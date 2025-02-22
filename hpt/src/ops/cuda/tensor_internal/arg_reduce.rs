use crate::ops::cuda::cuda_utils::{check_launch_config, max_grid_dim_y};
use crate::ops::cuda::utils::reduce::reduce_utils::rearrange_array;
use crate::ops::cuda::utils::unary::strided_copy::arg_reduce_strided_copy;
use crate::{
    ops::cuda::cuda_utils::{compute_kernel_launch_config, load_ptx_and_get_data},
    tensor_base::_Tensor,
    Cuda,
};
use cudarc::driver::{DeviceRepr, LaunchAsync};
use hpt_common::error::shape::ShapeError;
use hpt_common::{
    axis::axis::{process_axes, Axis},
    error::base::TensorError,
};
use hpt_cudakernels::{RegisterInfo, ARGMAX, ARGMIN};
use hpt_traits::{CommonBounds, IndexReduce, ShapeManipulate, TensorCreator, TensorInfo};
use hpt_types::dtype::CudaType;
use hpt_types::{
    into_scalar::Cast,
    type_promote::{Cmp, NormalOut},
};

#[track_caller]
pub(crate) fn arg_reduce<T, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    keepdims: bool,
    is_contiguous: bool,
    meta: &phf::Map<
        usize,
        (
            &'static str,
            &'static phf::Map<&'static str, RegisterInfo>,
            &'static [&str],
        ),
    >,
    module_name: &str,
    c: Option<_Tensor<i64, Cuda, DEVICE_ID>>,
) -> Result<_Tensor<i64, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + Cast<i64> + DeviceRepr + CudaType,
{
    let mut transposed_axis = rearrange_array(a.ndim(), axes);

    transposed_axis[..a.layout.ndim() - axes.len()]
        .sort_by(|x, y| a.layout.strides()[*y].cmp(&a.layout.strides()[*x]));
    transposed_axis[a.layout.ndim() - axes.len()..]
        .sort_by(|x, y| a.layout.strides()[*y].cmp(&a.layout.strides()[*x]));

    let res_layout = a.layout.reduce(axes, false)?;

    let mut result = if let Some(out) = c {
        ShapeError::check_inplace_out_layout_valid(res_layout.shape(), &out.layout())?;
        out
    } else {
        _Tensor::<i64, Cuda, DEVICE_ID>::empty(res_layout.shape())?
    };
    if a.ndim() == axes.len() {
        let res = result.cuda_slice();
        let (reduce_kernel, reg_info) = load_ptx_and_get_data(
            module_name,
            &if is_contiguous {
                format!("contiguous_reduce_{}", T::STR)
            } else {
                format!("uncontiguous_reduce_{}", T::STR)
            },
            a.device(),
            a.device_cap(),
            &meta,
        )
        .unwrap();
        let size = a.size();

        let compute_cfg = |reduce_size: usize| {
            let mut cfg = compute_kernel_launch_config(a.device(), &reg_info, reduce_size);
            let block_dim_x = cfg.block_dim.0.next_power_of_two().max(64);
            cfg.block_dim = (block_dim_x, 1, 1);
            // calculate the number of blocks needed, divide by 2 for reduction
            cfg.grid_dim = (
                reduce_size
                    .div_ceil(block_dim_x as usize)
                    .div_ceil(2)
                    .max(1) as u32,
                1,
                1,
            );
            cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
            check_launch_config(a.device(), &cfg).unwrap();
            cfg
        };

        let cfg = compute_cfg(size);
        let mut reduce_size = cfg.grid_dim.0 as usize;
        let tmp_buffer = unsafe { a.device().alloc::<T>(reduce_size).unwrap() };
        let tmp_buffer_idx = unsafe { a.device().alloc::<i64>(reduce_size).unwrap() };
        if is_contiguous {
            unsafe {
                reduce_kernel
                    .clone()
                    .launch(cfg, (&tmp_buffer, &tmp_buffer_idx, a.cuda_slice(), size))
            }
            .unwrap();
        } else {
            let cuda_shape = a.cuda_shape().unwrap();
            let cuda_strides = a.cuda_strides().unwrap();
            unsafe {
                reduce_kernel.clone().launch(
                    cfg,
                    (
                        &tmp_buffer,
                        &tmp_buffer_idx,
                        a.cuda_slice(),
                        &cuda_shape,
                        &cuda_strides,
                        a.ndim(),
                        size,
                    ),
                )
            }
            .unwrap();
        }

        let (reduce_kernel, _) = load_ptx_and_get_data(
            module_name,
            &format!("contiguous_reduce2_{}", T::STR),
            a.device(),
            a.device_cap(),
            &meta,
        )
        .unwrap();

        // keep reducing until the size is 1
        while reduce_size > 1 {
            let cfg = compute_cfg(reduce_size);
            // it is safe to use tmp_buffer as input and output because there is no data racing
            unsafe {
                reduce_kernel.clone().launch(
                    cfg,
                    (&tmp_buffer, &tmp_buffer_idx, &tmp_buffer, reduce_size),
                )
            }
            .unwrap();
            reduce_size = cfg.grid_dim.0 as usize;
        }
        a.device().synchronize().unwrap();
        let mut _res_ptr = unsafe { a.device().upgrade_device_ptr::<i64>(res.inner, 1) };
        a.device()
            .dtod_copy(&tmp_buffer_idx, &mut _res_ptr)
            .unwrap();
        _res_ptr.leak();
    } else {
        let mut perm = (0..a.ndim()).collect::<Vec<_>>();
        perm.remove(axes[0]);
        perm.push(axes[0]);
        let transposed_tensor = a.permute(&perm).unwrap();
        let inner_loop_size = *transposed_tensor.shape().last().unwrap() as usize;
        let outer_loop_size = a.size() / inner_loop_size;
        let (reduce_kernel, reg_info) = load_ptx_and_get_data(
            module_name,
            &format!("nkd_{}", T::STR),
            a.device(),
            a.device_cap(),
            &meta,
        )
        .unwrap();
        let mut reduce_size = inner_loop_size;

        let max_grid_dim_y = max_grid_dim_y(a.device());
        let compute_cfg = |reduce_size: usize| {
            let mut cfg = compute_kernel_launch_config(a.device(), &reg_info, reduce_size);
            let block_dim_x = cfg.block_dim.0.next_power_of_two().max(64);
            cfg.block_dim = (block_dim_x, 1, 1);
            // calculate the number of blocks needed, divide by 2 for reduction
            cfg.grid_dim = (
                reduce_size
                    .div_ceil(block_dim_x as usize)
                    .div_ceil(2)
                    .max(1) as u32,
                (outer_loop_size as u32).min(max_grid_dim_y),
                1,
            );
            cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
            check_launch_config(a.device(), &cfg).unwrap();
            cfg
        };

        let cfg = compute_cfg(reduce_size);
        let shape = transposed_tensor.cuda_shape().unwrap();
        let strides = transposed_tensor.cuda_strides().unwrap();
        let tmp_buffer = unsafe {
            a.device()
                .alloc::<T>(cfg.grid_dim.0 as usize * outer_loop_size)
                .unwrap()
        };
        let tmp_buffer_idx = unsafe {
            a.device()
                .alloc::<i64>(cfg.grid_dim.0 as usize * outer_loop_size)
                .unwrap()
        };
        for idx in (0..outer_loop_size).step_by(max_grid_dim_y as usize) {
            unsafe {
                reduce_kernel.clone().launch(
                    cfg,
                    (
                        &tmp_buffer,
                        transposed_tensor.cuda_slice(),
                        &tmp_buffer_idx,
                        &shape,
                        &strides,
                        transposed_tensor.ndim(),
                        idx,
                        reduce_size,
                        outer_loop_size,
                        cfg.grid_dim.0 as usize,
                    ),
                )
            }
            .unwrap();
        }

        reduce_size = cfg.grid_dim.0 as usize;
        let original_reduce_size = reduce_size;
        let reduce_kernel = a
            .device()
            .get_func(&module_name, &format!("nkd2_{}", T::STR))
            .unwrap();
        while reduce_size > 1 {
            let cfg = compute_cfg(reduce_size);
            for idx in (0..outer_loop_size).step_by(max_grid_dim_y as usize) {
                unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            &tmp_buffer,
                            &tmp_buffer,
                            &tmp_buffer_idx,
                            idx,
                            reduce_size,
                            outer_loop_size,
                            cfg.grid_dim.0 as usize,
                            original_reduce_size,
                        ),
                    )
                }
                .unwrap();
            }
            reduce_size = cfg.grid_dim.0 as usize;
        }
        a.device().synchronize().unwrap();
        arg_reduce_strided_copy(&tmp_buffer_idx, &mut result.clone(), original_reduce_size)
            .expect("strided_copy failed");
    }
    if keepdims {
        let res_layout = a.layout.reduce(axes, true)?;
        result = result.reshape(res_layout.shape())?;
    }
    Ok(result)
}

impl<
        T: CommonBounds + NormalOut<Output = T> + Cmp + DeviceRepr + CudaType + Cast<i64>,
        const DEVICE_ID: usize,
    > IndexReduce for _Tensor<T, Cuda, DEVICE_ID>
{
    type Output = _Tensor<i64, Cuda, DEVICE_ID>;

    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        let axis: Axis = axis.into();
        let axes: Vec<usize> = process_axes(axis.clone(), self.ndim())?;
        arg_reduce(
            self,
            &axes,
            keep_dims,
            self.is_contiguous() && self.parent().is_none(),
            &ARGMAX,
            "argmax",
            None,
        )
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        let axis: Axis = axis.into();
        let axes: Vec<usize> = process_axes(axis.clone(), self.ndim())?;
        arg_reduce(
            self,
            &axes,
            keep_dims,
            self.is_contiguous() && self.parent().is_none(),
            &ARGMIN,
            "argmin",
            None,
        )
    }
}
