use crate::{
    ops::cuda::{
        cuda_utils::{compute_kernel_launch_config, load_ptx_and_get_data},
        utils::reduce::reduce_template::contiguous_reduce_template,
    },
    tensor_base::_Tensor,
    Cuda,
};
use cudarc::driver::LaunchConfig;
use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync};
use hpt_common::{
    axis::axis::{process_axes, Axis},
    error::base::TensorError,
};
use hpt_cudakernels::{RegisterInfo, ARGMAX, ARGMIN};
use hpt_traits::{CommonBounds, IndexReduce, ShapeManipulate, TensorInfo};
use hpt_types::dtype::CudaType;
use hpt_types::{
    into_scalar::Cast,
    type_promote::{Cmp, NormalOut},
};

#[track_caller]
pub(crate) fn contiguous_reduce<T, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    keepdims: bool,
    init_out: bool,
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
    let max_axis = *axes.iter().max().unwrap();
    let (a, fused_dims) = if max_axis == a.ndim() - 1 {
        (a.clone(), vec![])
    } else {
        let prod = a.shape()[max_axis + 1..].iter().product::<i64>();
        let new_shape: Vec<i64> = a.shape()[..=max_axis].to_vec();
        let mut new_shape = new_shape;
        new_shape.push(prod);
        (a.reshape(&new_shape)?, a.shape()[max_axis + 1..].to_vec())
    };
    let res: _Tensor<i64, Cuda, DEVICE_ID> = contiguous_reduce_template(
        &a,
        axes,
        0i64,
        keepdims,
        init_out,
        c,
        |res| {
            let (reduce_kernel, reg_info) = load_ptx_and_get_data(
                module_name,
                &format!("contiguous_reduce_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta,
            )
            .unwrap();
            let mut size = a.size();
            let mut cfg = compute_kernel_launch_config(a.device(), &reg_info, size);
            cfg.block_dim.0 = (2f64.powf((cfg.block_dim.0 as f64).log2().floor()) as u32).max(64);
            let grid_dim = (size as u32 + cfg.block_dim.0 - 1) / cfg.block_dim.0;
            cfg.grid_dim.0 = ((grid_dim + 2 - 1) / 2).max(1);
            let mut num_blocks = cfg.grid_dim.0;
            cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
            let mut tmp_res = unsafe { a.device().alloc::<T>(num_blocks as usize).unwrap() };
            let mut tmp_idx = unsafe { a.device().alloc::<i64>(num_blocks as usize).unwrap() };
            unsafe {
                reduce_kernel
                    .clone()
                    .launch(cfg, (&mut tmp_res, &mut tmp_idx, a.cuda_slice(), size))
            }
            .unwrap();
            size = num_blocks as usize;
            let mut inp = tmp_res;
            let mut inp_idx = tmp_idx;

            let reduce_kernel = a
                .device()
                .get_func(module_name, &format!("contiguous_reduce2_{}", T::STR))
                .unwrap();

            while num_blocks > 1 {
                cfg = compute_kernel_launch_config(a.device(), &reg_info, size);
                cfg.block_dim.0 =
                    (2f64.powf((cfg.block_dim.0 as f64).log2().floor()) as u32).max(64);
                let grid_dim = (size as u32 + cfg.block_dim.0 - 1) / cfg.block_dim.0;
                cfg.grid_dim.0 = ((grid_dim + 2 - 1) / 2).max(1);
                num_blocks = cfg.grid_dim.0;
                cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                    + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
                let mut tmp_res = unsafe { a.device().alloc::<T>(num_blocks as usize).unwrap() };
                let mut tmp_idx = unsafe { a.device().alloc::<i64>(num_blocks as usize).unwrap() };
                unsafe {
                    reduce_kernel
                        .clone()
                        .launch(cfg, (&mut tmp_res, &mut tmp_idx, &inp, &inp_idx, size))
                }
                .unwrap();
                inp = tmp_res;
                inp_idx = tmp_idx;
                size = num_blocks as usize;
            }
            a.device().synchronize().unwrap();
            assert_eq!(inp.len(), 1);
            let mut _res_ptr = unsafe { a.device().upgrade_device_ptr::<i64>(res.inner, 1) };
            a.device().dtod_copy(&inp_idx, &mut _res_ptr).unwrap();
            _res_ptr.leak();
        },
        |inner_loop_size, inner_loop_size2, res, transposed_tensor| {
            let outer_loop_size = a.size() / (inner_loop_size * inner_loop_size2);
            let (reduce_kernel, reg_info) = load_ptx_and_get_data(
                module_name,
                &format!("contiguous_reduce3_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta,
            )
            .unwrap();
            let mut reduce_size = inner_loop_size * inner_loop_size2;
            let mut cfg = compute_kernel_launch_config(a.device(), &reg_info, reduce_size);
            cfg.block_dim.0 = (2f64.powf((cfg.block_dim.0 as f64).log2().floor()) as u32).max(64);
            let grid_dim = (reduce_size as u32 + cfg.block_dim.0 - 1) / cfg.block_dim.0;
            cfg.grid_dim.0 = ((grid_dim + 1) / 2).max(1);
            cfg.grid_dim.1 = outer_loop_size as u32;

            cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let mut tmp_res = unsafe {
                a.device()
                    .alloc::<T>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
            let mut tmp_idx = unsafe {
                a.device()
                    .alloc::<i64>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
            unsafe {
                reduce_kernel.clone().launch(
                    cfg,
                    (
                        &mut tmp_res,
                        &mut tmp_idx,
                        transposed_tensor.cuda_slice(),
                        &shape,
                        &strides,
                        transposed_tensor.ndim(),
                        reduce_size,
                        cfg.grid_dim.0 as usize,
                    ),
                )
            }
            .unwrap();
            reduce_size = cfg.grid_dim.0 as usize;
            let mut inp = tmp_res;
            let mut inp_idx = tmp_idx;
            let reduce_kernel = a
                .device()
                .get_func(&module_name, &format!("contiguous_reduce33_{}", T::STR))
                .unwrap();
            while reduce_size > 1 {
                cfg = compute_kernel_launch_config(a.device(), &reg_info, reduce_size);
                cfg.block_dim.0 =
                    (2f64.powf((cfg.block_dim.0 as f64).log2().floor()) as u32).max(64);
                let grid_dim = (reduce_size as u32 + cfg.block_dim.0 - 1) / cfg.block_dim.0;
                cfg.grid_dim.0 = ((grid_dim + 1) / 2).max(1);
                cfg.grid_dim.1 = outer_loop_size as u32;
                cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                    + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
                let mut tmp_res = unsafe {
                    a.device()
                        .alloc::<T>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                        .unwrap()
                };
                let mut tmp_idx = unsafe {
                    a.device()
                        .alloc::<i64>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                        .unwrap()
                };
                unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            &mut tmp_res,
                            &mut tmp_idx,
                            &inp,
                            &inp_idx,
                            reduce_size,
                            cfg.grid_dim.0 as usize,
                        ),
                    )
                }
                .unwrap();
                inp = tmp_res;
                inp_idx = tmp_idx;
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            let mut _res_ptr = unsafe {
                a.device()
                    .upgrade_device_ptr::<i64>(res.cuda_slice().inner, res.size())
            };
            a.device().dtod_copy(&inp_idx, &mut _res_ptr).unwrap();
            _res_ptr.leak();
        },
        |inner_loop_size_2, result, transposed_tensor| {
            let perm = (0..transposed_tensor.ndim()).collect::<Vec<_>>();
            let right = perm[perm.len() - axes.len()..].to_vec();
            let left = perm[..perm.len() - axes.len()].to_vec();
            let mut perm = right;
            perm.extend(left);
            let transposed_tensor = transposed_tensor.permute(&perm).unwrap();
            let (reduce_kernel, _) = load_ptx_and_get_data(
                module_name,
                &format!("contiguous_reduce4_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta,
            )
            .unwrap();
            let block_dim_x = 32;
            let block_dim_y = 32;
            let mut reduce_size = inner_loop_size_2;
            let grid_dim_x = (a.size() / reduce_size).div_ceil(block_dim_x as usize) as u32;
            let launch_cfg = LaunchConfig {
                block_dim: (block_dim_x, block_dim_y, 1),
                grid_dim: (
                    grid_dim_x,
                    reduce_size.div_ceil(block_dim_y as usize) as u32,
                    1,
                ),
                shared_mem_bytes: block_dim_x * block_dim_y * std::mem::size_of::<T>() as u32
                    + block_dim_x * block_dim_y * std::mem::size_of::<i64>() as u32,
            };
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let mut height = launch_cfg.grid_dim.1;
            let mut tmp_res = unsafe {
                a.device()
                    .alloc::<T>(result.size() * height as usize)
                    .unwrap()
            };
            let mut tmp_idx = unsafe {
                a.device()
                    .alloc::<i64>(result.size() * height as usize)
                    .unwrap()
            };
            unsafe {
                reduce_kernel.launch(
                    launch_cfg,
                    (
                        &mut tmp_res,
                        &mut tmp_idx,
                        transposed_tensor.cuda_slice(),
                        &shape,
                        &strides,
                        transposed_tensor.ndim(),
                        result.size(),
                        reduce_size,
                    ),
                )
            }
            .unwrap();
            let (reduce_kernel, _) = load_ptx_and_get_data(
                module_name,
                &format!("contiguous_reduce44_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta,
            )
            .unwrap();
            let mut inp = tmp_res;
            let mut inp_idx = tmp_idx;
            while height > 1 {
                reduce_size = height as usize;
                let launch_cfg = LaunchConfig {
                    block_dim: (block_dim_x, block_dim_y, 1),
                    grid_dim: (
                        grid_dim_x,
                        reduce_size.div_ceil(block_dim_y as usize) as u32,
                        1,
                    ),
                    shared_mem_bytes: block_dim_x * block_dim_y * std::mem::size_of::<T>() as u32
                        + block_dim_x * block_dim_y * std::mem::size_of::<i64>() as u32,
                };
                let mut tmp_res = unsafe {
                    a.device()
                        .alloc::<T>(result.size() * launch_cfg.grid_dim.1 as usize)
                        .unwrap()
                };
                let mut tmp_idx = unsafe {
                    a.device()
                        .alloc::<i64>(result.size() * launch_cfg.grid_dim.1 as usize)
                        .unwrap()
                };
                unsafe {
                    reduce_kernel.clone().launch(
                        launch_cfg,
                        (
                            &mut tmp_res,
                            &mut tmp_idx,
                            &inp,
                            &inp_idx,
                            transposed_tensor.ndim(),
                            result.size(),
                            reduce_size,
                        ),
                    )
                }
                .unwrap();
                inp = tmp_res;
                inp_idx = tmp_idx;
                height = launch_cfg.grid_dim.1;
            }
            a.device().synchronize().unwrap();
            let mut _res_ptr = unsafe {
                a.device()
                    .upgrade_device_ptr::<i64>(result.cuda_slice().inner, result.size())
            };
            a.device().dtod_copy(&inp_idx, &mut _res_ptr).unwrap();
            _res_ptr.leak();
        },
    )?;
    if !fused_dims.is_empty() {
        let res_shape = res.shape().clone();
        let mut new_shape = res_shape.clone();
        new_shape.pop();
        new_shape.extend(fused_dims.iter());
        res.reshape(&new_shape)
    } else {
        Ok(res)
    }
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
        contiguous_reduce(self, &axes, keep_dims, false, &ARGMAX, "argmax", None)
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        let axis: Axis = axis.into();
        let axes: Vec<usize> = process_axes(axis.clone(), self.ndim())?;
        contiguous_reduce(self, &axes, keep_dims, false, &ARGMIN, "argmin", None)
    }
}
