use crate::ops::cuda::utils::unary::strided_copy::strided_copy;
use crate::{
    ops::cuda::{
        cuda_utils::{compute_kernel_launch_config, load_ptx_and_get_data},
        utils::reduce::reduce_template::{contiguous_reduce_template, uncontiguos_reduce_template},
    },
    tensor_base::_Tensor,
    Cuda,
};
use cudarc::driver::{DeviceRepr, LaunchAsync};
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
                cfg
            };

            let cfg = compute_cfg(size);
            let mut reduce_size = cfg.grid_dim.0 as usize;
            let tmp_buffer = unsafe { a.device().alloc::<T>(reduce_size).unwrap() };
            let tmp_buffer_idx = unsafe { a.device().alloc::<i64>(reduce_size).unwrap() };
            unsafe {
                reduce_kernel
                    .clone()
                    .launch(cfg, (&tmp_buffer, &tmp_buffer_idx, a.cuda_slice(), size))
            }
            .unwrap();

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
        },
        |inner_loop_size, inner_loop_size2, res, transposed_tensor| {
            assert_eq!(inner_loop_size2, 1);
            let outer_loop_size = a.size() / inner_loop_size;
            let (reduce_kernel, reg_info) = load_ptx_and_get_data(
                module_name,
                &format!("nkd_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta,
            )
            .unwrap();
            let mut reduce_size = inner_loop_size * inner_loop_size2;

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
                    outer_loop_size as u32,
                    1,
                );
                cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                    + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
                cfg
            };

            let cfg = compute_cfg(reduce_size);
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let tmp_buffer = unsafe {
                a.device()
                    .alloc::<T>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
            let tmp_buffer_idx = unsafe {
                a.device()
                    .alloc::<i64>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
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
                        reduce_size,
                        cfg.grid_dim.0 as usize,
                    ),
                )
            }
            .unwrap();

            reduce_size = cfg.grid_dim.0 as usize;
            let reduce_kernel = a
                .device()
                .get_func(&module_name, &format!("nkd2_{}", T::STR))
                .unwrap();
            while reduce_size > 1 {
                let cfg = compute_cfg(reduce_size);
                unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            &tmp_buffer,
                            &tmp_buffer,
                            &tmp_buffer_idx,
                            reduce_size,
                            cfg.grid_dim.0 as usize,
                        ),
                    )
                }
                .unwrap();
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            let mut _res_ptr = unsafe {
                a.device()
                    .upgrade_device_ptr::<i64>(res.cuda_slice().inner, res.size())
            };
            a.device()
                .dtod_copy(&tmp_buffer_idx, &mut _res_ptr)
                .unwrap();
            _res_ptr.leak();
        },
        |_, result, transposed_tensor| {
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
                    outer_loop_size as u32,
                    1,
                );
                cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                    + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
                cfg
            };

            let cfg = compute_cfg(reduce_size);
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let tmp_buffer = unsafe {
                a.device()
                    .alloc::<T>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
            let tmp_buffer_idx = unsafe {
                a.device()
                    .alloc::<i64>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
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
                        reduce_size,
                        cfg.grid_dim.0 as usize,
                    ),
                )
            }
            .unwrap();

            reduce_size = cfg.grid_dim.0 as usize;
            let reduce_kernel = a
                .device()
                .get_func(&module_name, &format!("nkd2_{}", T::STR))
                .unwrap();
            while reduce_size > 1 {
                let cfg = compute_cfg(reduce_size);
                unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            &tmp_buffer,
                            &tmp_buffer,
                            &tmp_buffer_idx,
                            reduce_size,
                            cfg.grid_dim.0 as usize,
                        ),
                    )
                }
                .unwrap();
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            let mut _res_ptr = unsafe {
                a.device()
                    .upgrade_device_ptr::<i64>(result.cuda_slice().inner, result.size())
            };
            a.device()
                .dtod_copy(&tmp_buffer_idx, &mut _res_ptr)
                .unwrap();
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

#[track_caller]
pub(crate) fn uncontiguous_reduce<T, const DEVICE_ID: usize>(
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
) -> std::result::Result<_Tensor<i64, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType + Cast<i64>,
{
    uncontiguos_reduce_template(
        a,
        axes,
        0i64,
        keepdims,
        init_out,
        c,
        |res| {
            let (reduce_kernel, reg_info) = load_ptx_and_get_data(
                module_name,
                &format!("uncontiguous_reduce_{}", T::STR),
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
                cfg
            };

            let cfg = compute_cfg(size);
            let mut reduce_size = cfg.grid_dim.0 as usize;
            let tmp_buffer = unsafe { a.device().alloc::<T>(reduce_size).unwrap() };
            let tmp_buffer_idx = unsafe { a.device().alloc::<i64>(reduce_size).unwrap() };
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
        },
        |_, inner_loop_size, inner_loop_size2, res, transposed_tensor| {
            assert_eq!(inner_loop_size2, 1);
            let outer_loop_size = a.size() / inner_loop_size;
            let (reduce_kernel, reg_info) = load_ptx_and_get_data(
                module_name,
                &format!("nkd_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta,
            )
            .unwrap();
            let mut reduce_size = inner_loop_size * inner_loop_size2;

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
                    outer_loop_size as u32,
                    1,
                );
                cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                    + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
                cfg
            };

            let cfg = compute_cfg(reduce_size);
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let tmp_buffer = unsafe {
                a.device()
                    .alloc::<T>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
            let tmp_buffer_idx = unsafe {
                a.device()
                    .alloc::<i64>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
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
                        reduce_size,
                        cfg.grid_dim.0 as usize,
                    ),
                )
            }
            .unwrap();

            reduce_size = cfg.grid_dim.0 as usize;
            let reduce_kernel = a
                .device()
                .get_func(&module_name, &format!("nkd2_{}", T::STR))
                .unwrap();
            while reduce_size > 1 {
                let cfg = compute_cfg(reduce_size);
                unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            &tmp_buffer,
                            &tmp_buffer,
                            &tmp_buffer_idx,
                            reduce_size,
                            cfg.grid_dim.0 as usize,
                        ),
                    )
                }
                .unwrap();
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            strided_copy(&tmp_buffer_idx, &mut res.clone()).expect("strided_copy failed");
        },
        |_, _, _, result, transposed_tensor| {
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
                    outer_loop_size as u32,
                    1,
                );
                cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32
                    + cfg.block_dim.0 * std::mem::size_of::<i64>() as u32;
                cfg
            };

            let cfg = compute_cfg(reduce_size);
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let tmp_buffer = unsafe {
                a.device()
                    .alloc::<T>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
            let tmp_buffer_idx = unsafe {
                a.device()
                    .alloc::<i64>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
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
                        reduce_size,
                        cfg.grid_dim.0 as usize,
                    ),
                )
            }
            .unwrap();

            reduce_size = cfg.grid_dim.0 as usize;
            let reduce_kernel = a
                .device()
                .get_func(&module_name, &format!("nkd2_{}", T::STR))
                .unwrap();
            while reduce_size > 1 {
                let cfg = compute_cfg(reduce_size);
                unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            &tmp_buffer,
                            &tmp_buffer,
                            &tmp_buffer_idx,
                            reduce_size,
                            cfg.grid_dim.0 as usize,
                        ),
                    )
                }
                .unwrap();
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            strided_copy(&tmp_buffer_idx, &mut result.clone()).expect("strided_copy failed");
        },
    )
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
        if self.is_contiguous() && self.parent().is_none() {
            contiguous_reduce(self, &axes, keep_dims, false, &ARGMAX, "argmax", None)
        } else {
            uncontiguous_reduce(self, &axes, keep_dims, false, &ARGMAX, "argmax", None)
        }
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        let axis: Axis = axis.into();
        let axes: Vec<usize> = process_axes(axis.clone(), self.ndim())?;
        if self.is_contiguous() && self.parent().is_none() {
            contiguous_reduce(self, &axes, keep_dims, false, &ARGMIN, "argmin", None)
        } else {
            uncontiguous_reduce(self, &axes, keep_dims, false, &ARGMIN, "argmin", None)
        }
    }
}
