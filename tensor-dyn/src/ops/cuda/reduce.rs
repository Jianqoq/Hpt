#![allow(unused)]

use crate::ops::cuda::reduce_template::contiguous_reduce_template;
use crate::tensor_base::_Tensor;
use crate::Cuda;

use super::cuda_utils::compute_kernel_launch_config;
use super::cuda_utils::load_ptx_and_get_data;
use super::reduce_template::uncontiguos_reduce_template;
use anyhow;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;
use cudarc::types::CudaTypeName;
use tensor_common::err_handler::TensorError;
use tensor_cudakernels::RegisterInfo;
use tensor_traits::shape_manipulate::ShapeManipulate;
use tensor_traits::tensor::CommonBounds;
use tensor_traits::tensor::TensorInfo;
use tensor_traits::TensorCreator;
use tensor_traits::TensorLike;
use tensor_types::convertion::Convertor;
use tensor_types::cast::Cast;

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn reduce<T, F, F2, F3, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    op: F,
    op_no_cast: F2,
    vec_op: F3,
    axes: &[usize],
    init_val: T,
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
    c: Option<_Tensor<T, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<T, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + Cast<T> + Convertor + DeviceRepr + CudaTypeName,
    F: Fn(T, T) -> T + Sync + Send + 'static + Copy,
    F2: Fn(T, T) -> T + Sync + Send + 'static + Copy,
    F3: Fn(T::Vec, T::Vec) -> T::Vec + Sync + Send + 'static + Copy,
    T::Vec: Copy,
{
    if a.is_contiguous() && a.parent().is_none() {
        contiguous_reduce::<_, _, _, _, fn(T) -> T, _, _, fn(T::Vec) -> T::Vec, T, DEVICE_ID>(
            a,
            op,
            op_no_cast,
            op,
            None,
            vec_op,
            vec_op,
            None,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            c,
        )
    } else {
        uncontiguous_reduce::<_, _, _, fn(T) -> T, _, fn(T::Vec) -> T::Vec, T, DEVICE_ID>(
            a, op, op, None, vec_op, None, &axes, init_val, keepdims, init_out, c,
        )
    }
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn reduce2<T, F, F2, F3, F4, F5, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    op: F,
    op_no_cast: F2,
    op2: F3,
    vec_op: F4,
    vec_op2: F5,
    axes: &[usize],
    init_val: O,
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
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + Cast<O> + Convertor + DeviceRepr + CudaTypeName,
    F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(O::Vec, T::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    F5: Fn(O::Vec, O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    O: CommonBounds + DeviceRepr + CudaTypeName,
    T::Vec: Copy,
    O::Vec: Copy,
{
    if a.is_contiguous() && a.parent().is_none() {
        contiguous_reduce::<T, F, F2, F3, fn(O) -> O, _, _, fn(O::Vec) -> O::Vec, O, DEVICE_ID>(
            a,
            op,
            op_no_cast,
            op2,
            None,
            vec_op,
            vec_op2,
            None,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            c,
        )
    } else {
        uncontiguous_reduce::<T, F, F3, fn(O) -> O, _, fn(O::Vec) -> O::Vec, O, DEVICE_ID>(
            a, op, op2, None, vec_op, None, &axes, init_val, keepdims, init_out, c,
        )
    }
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn reduce3<T, F, F2, F3, F4, F5, F6, F7, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    op: F,
    op_no_cast: F2,
    op2: F3,
    op3: F4,
    vec_op: F5,
    vec_op2: F6,
    op5: F7,
    axes: &[usize],
    init_val: O,
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
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + Cast<O> + Convertor + DeviceRepr + CudaTypeName,
    F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(O) -> O + Sync + Send + 'static + Copy,
    F5: Fn(O::Vec, T::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    F6: Fn(O::Vec, O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    F7: Fn(O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    O: CommonBounds + DeviceRepr + CudaTypeName,
    O::Vec: Copy,
{
    if a.is_contiguous() && a.parent().is_none() {
        contiguous_reduce::<T, F, F2, F3, F4, F5, F6, F7, O, DEVICE_ID>(
            a,
            op,
            op_no_cast,
            op2,
            Some(op3),
            vec_op,
            vec_op2,
            Some(op5),
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            c,
        )
    } else {
        uncontiguous_reduce::<T, F, F3, F4, F5, F7, O, DEVICE_ID>(
            a,
            op,
            op2,
            Some(op3),
            vec_op,
            Some(op5),
            &axes,
            init_val,
            keepdims,
            init_out,
            c,
        )
    }
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn contiguous_reduce<T, F, F2, F3, F4, F5, F6, F7, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    op: F,
    op_no_cast: F2,
    op2: F3,
    op3: Option<F4>,
    vec_op: F5,
    vec_op2: F6,
    vec_post: Option<F7>,
    axes: &[usize],
    init_val: O,
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
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + Cast<O> + Convertor + DeviceRepr + CudaTypeName,
    O: CommonBounds + DeviceRepr + CudaTypeName,
    F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(O) -> O + Sync + Send + 'static + Copy,
    F5: Fn(O::Vec, T::Vec) -> O::Vec + 'static + Copy + Send + std::marker::Sync,
    F6: Fn(O::Vec, O::Vec) -> O::Vec + 'static + Copy + Send + std::marker::Sync,
    F7: Fn(O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    T::Vec: Copy,
    O::Vec: Copy,
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
    let res = contiguous_reduce_template(
        &a,
        axes,
        init_val,
        keepdims,
        init_out,
        c,
        |res| {
            let (reduce_kernel, reg_info) = load_ptx_and_get_data(
                module_name,
                &format!("contiguous_reduce_{}", T::ID),
                a.device(),
                a.device_cap(),
                &meta,
            )
            .unwrap();
            let mut size = a.size();

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
                cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32;
                cfg
            };

            let cfg = compute_cfg(size);
            let mut reduce_size = cfg.grid_dim.0 as usize;
            let tmp_res = unsafe { a.device().alloc::<T>(reduce_size).unwrap() };
            let tmp_buffer = tmp_res.leak();
            unsafe {
                reduce_kernel
                    .clone()
                    .launch(cfg, (tmp_buffer, a.cuda_slice(), size))
            }
            .unwrap();

            // keep reducing until the size is 1
            while reduce_size > 1 {
                let cfg = compute_cfg(reduce_size);
                // it is safe to use tmp_buffer as input and output because there is no data racing
                unsafe {
                    reduce_kernel
                        .clone()
                        .launch(cfg, (tmp_buffer, tmp_buffer, reduce_size))
                }
                .unwrap();
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            let tmp_buffer = unsafe { a.device().upgrade_device_ptr::<O>(tmp_buffer, 1) };
            let mut _res_ptr = unsafe { a.device().upgrade_device_ptr::<O>(res.inner, 1) };
            a.device().dtod_copy(&tmp_buffer, &mut _res_ptr).unwrap();
            _res_ptr.leak();
        },
        |inner_loop_size, inner_loop_size2, res, transposed_tensor| {
            let outer_loop_size = a.size() / (inner_loop_size * inner_loop_size2);
            let (reduce_kernel, reg_info) = load_ptx_and_get_data(
                module_name,
                &format!("contiguous_reduce2_{}", T::ID),
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
                cfg.shared_mem_bytes = cfg.block_dim.0 * std::mem::size_of::<T>() as u32;
                cfg
            };

            let cfg = compute_cfg(reduce_size);
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let mut tmp_res = unsafe {
                a.device()
                    .alloc::<T>((cfg.grid_dim.0 * cfg.grid_dim.1) as usize)
                    .unwrap()
            };
            let tmp_buffer = tmp_res.leak();
            unsafe {
                reduce_kernel.clone().launch(
                    cfg,
                    (
                        tmp_buffer,
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
            let mut inp = tmp_buffer;
            let reduce_kernel = a
                .device()
                .get_func(&module_name, &format!("contiguous_reduce22_{}", T::ID))
                .unwrap();
            while reduce_size > 1 {
                let cfg = compute_cfg(reduce_size);
                unsafe {
                    reduce_kernel
                        .clone()
                        .launch(cfg, (tmp_buffer, inp, reduce_size, cfg.grid_dim.0 as usize))
                }
                .unwrap();
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            let mut _res_ptr = unsafe {
                a.device()
                    .upgrade_device_ptr::<O>(res.cuda_slice().inner, res.size())
            };
            let tmp_buffer = unsafe { a.device().upgrade_device_ptr::<O>(tmp_buffer, res.size()) };
            a.device().dtod_copy(&tmp_buffer, &mut _res_ptr).unwrap();
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
                &format!("contiguous_reduce3_{}", T::ID),
                a.device(),
                a.device_cap(),
                &meta,
            )
            .unwrap();
            let block_dim_x = 32; // x dimension must be 32, as it is the warp size
            let block_dim_y = 32; // y dimension depends on the register used, currently 32
            let mut reduce_size = inner_loop_size_2;
            let grid_dim_x = (a.size() / reduce_size).div_ceil(block_dim_x as usize) as u32;
            let launch_cfg = LaunchConfig {
                block_dim: (block_dim_x, block_dim_y, 1),
                grid_dim: (
                    grid_dim_x,
                    reduce_size.div_ceil(block_dim_y as usize) as u32,
                    1,
                ),
                shared_mem_bytes: block_dim_x * block_dim_y * std::mem::size_of::<T>() as u32,
            };
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let mut height = launch_cfg.grid_dim.1;
            let mut tmp_res = unsafe {
                a.device()
                    .alloc::<T>(result.size() * height as usize)
                    .unwrap()
            };
            let tmp_buffer = tmp_res.leak();
            unsafe {
                reduce_kernel.launch(
                    launch_cfg,
                    (
                        tmp_buffer,
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
                &format!("contiguous_reduce33_{}", T::ID),
                a.device(),
                a.device_cap(),
                &meta,
            )
            .unwrap();
            while height > 1 {
                reduce_size = height as usize;
                let launch_cfg = LaunchConfig {
                    block_dim: (block_dim_x, block_dim_y, 1),
                    grid_dim: (
                        grid_dim_x,
                        reduce_size.div_ceil(block_dim_y as usize) as u32,
                        1,
                    ),
                    shared_mem_bytes: block_dim_x * block_dim_y * std::mem::size_of::<T>() as u32,
                };
                unsafe {
                    reduce_kernel.clone().launch(
                        launch_cfg,
                        (
                            tmp_buffer,
                            tmp_buffer,
                            transposed_tensor.ndim(),
                            result.size(),
                            reduce_size,
                        ),
                    )
                }
                .unwrap();
                height = launch_cfg.grid_dim.1;
            }
            a.device().synchronize().unwrap();
            let mut _res_ptr = unsafe {
                a.device()
                    .upgrade_device_ptr::<O>(result.cuda_slice().inner, result.size())
            };
            let tmp_buffer = unsafe {
                a.device()
                    .upgrade_device_ptr::<O>(tmp_buffer, result.size())
            };
            a.device().dtod_copy(&tmp_buffer, &mut _res_ptr).unwrap();
            _res_ptr.leak();
        },
    )?;
    if !fused_dims.is_empty() {
        let res_shape = res.shape().clone();
        let mut new_shape = res_shape.clone();
        new_shape.pop();
        new_shape.extend(fused_dims.iter());
        Ok(res.reshape(&new_shape)?)
    } else {
        Ok(res)
    }
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn uncontiguous_reduce<T, F, F2, F3, F4, F5, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    op: F,
    op2: F2,
    op3: Option<F3>,
    _: F4,
    _: Option<F5>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + Cast<O> + Convertor + DeviceRepr + CudaTypeName,
    O: CommonBounds + DeviceRepr + CudaTypeName,
    F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(O::Vec, T::Vec) -> O::Vec + 'static + Copy + Send + std::marker::Sync,
    F5: Fn(O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    T::Vec: Copy,
    O::Vec: Copy,
{
    uncontiguos_reduce_template(
        a,
        axes,
        init_val,
        keepdims,
        init_out,
        c,
        move |res| unimplemented!(),
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| unimplemented!(),
        move |num_threads, inner_loop_size, ap, result| unimplemented!(),
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| unimplemented!(),
    )
}
