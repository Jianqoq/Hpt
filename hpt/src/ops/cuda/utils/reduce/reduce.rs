use crate::ops::cuda::cuda_utils::check_launch_config;
use crate::ops::cuda::cuda_utils::compute_num_blocks;
use crate::ops::cuda::cuda_utils::max_grid_dim_y;
use crate::ops::cuda::utils::unary::strided_copy::strided_copy;
use crate::ops::cuda::utils::unary::unary::unary_raw_mut;
use crate::tensor_base::_Tensor;
use crate::Cuda;

use crate::ops::cuda::cuda_utils::compute_kernel_launch_config;
use crate::ops::cuda::cuda_utils::load_ptx_and_get_data;
use crate::ops::cuda::utils::reduce::reduce_template::{
    contiguous_reduce_template,
    uncontiguos_reduce_template,
};
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;
use hpt_common::error::base::TensorError;
use hpt_cudakernels::RegisterInfo;
use hpt_traits::shape_manipulate::ShapeManipulate;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;

#[track_caller]
pub(crate) fn reduce<T, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    init_val: T,
    keepdims: bool,
    init_out: bool,
    meta: &phf::Map<
        usize,
        (&'static str, &'static phf::Map<&'static str, RegisterInfo>, &'static [&str])
    >,
    module_name: &str,
    op: &str,
    has_cumulate: bool,
    post_op: Option<impl Fn(Scalar<T>, Scalar<T>) -> Scalar<T>>,
    c: Option<_Tensor<T, Cuda, DEVICE_ID>>
) -> std::result::Result<_Tensor<T, Cuda, DEVICE_ID>, TensorError>
    where T: CommonBounds + Cast<T> + DeviceRepr + CudaType + Cast<f64>
{
    if a.is_contiguous() && a.parent().is_none() {
        contiguous_reduce::<T, T, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            has_cumulate,
            post_op,
            c
        )
    } else {
        uncontiguous_reduce::<T, T, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            has_cumulate,
            post_op,
            c
        )
    }
}

#[track_caller]
pub(crate) fn reduce2<T, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    meta: &phf::Map<
        usize,
        (&'static str, &'static phf::Map<&'static str, RegisterInfo>, &'static [&str])
    >,
    module_name: &str,
    op: &str,
    has_cumulate: bool,
    post_op: Option<impl Fn(Scalar<O>, Scalar<O>) -> Scalar<O>>,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>
)
    -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
    where
        T: CommonBounds + Cast<O> + DeviceRepr + CudaType + Cast<f64>,
        O: CommonBounds + DeviceRepr + CudaType,
        T::Vec: Copy,
        O::Vec: Copy
{
    if a.is_contiguous() && a.parent().is_none() {
        contiguous_reduce::<T, O, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            has_cumulate,
            post_op,
            c
        )
    } else {
        uncontiguous_reduce::<T, O, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            has_cumulate,
            post_op,
            c
        )
    }
}

#[track_caller]
pub(crate) fn reduce3<T, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    meta: &phf::Map<
        usize,
        (&'static str, &'static phf::Map<&'static str, RegisterInfo>, &'static [&str])
    >,
    module_name: &str,
    op: &str,
    has_cumulate: bool,
    post_op: Option<impl Fn(Scalar<O>, Scalar<O>) -> Scalar<O>>,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>
)
    -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
    where
        T: CommonBounds + Cast<O> + DeviceRepr + CudaType + Cast<f64>,
        O: CommonBounds + DeviceRepr + CudaType
{
    if a.is_contiguous() && a.parent().is_none() {
        contiguous_reduce::<T, O, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            has_cumulate,
            post_op,
            c
        )
    } else {
        uncontiguous_reduce::<T, O, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            has_cumulate,
            post_op,
            c
        )
    }
}

fn last_power_of_two(val: usize) -> usize {
    if val <= 1 {
        return val;
    }

    // Find the position of the highest bit set
    // This is equivalent to floor(log2(size))
    let highest_bit = usize::BITS - (val - 1).leading_zeros() - 1;

    // 2^highest_bit is the largest power of 2 <= size
    1 << highest_bit
}

/// Returns the largest multiple of `divisor` that is less than or equal to `value`
fn last_multiple_of(value: usize, divisor: usize) -> usize {
    if divisor == 0 {
        panic!("Division by zero in last_multiple_of");
    }

    // Integer division truncates toward zero, giving us the floor
    // of value/divisor, then multiply back to get the largest multiple
    (value / divisor) * divisor
}

#[track_caller]
pub(crate) fn contiguous_reduce<T, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    meta: &phf::Map<
        usize,
        (&'static str, &'static phf::Map<&'static str, RegisterInfo>, &'static [&str])
    >,
    module_name: &str,
    op: &str,
    has_cumulate: bool,
    post_op: Option<impl Fn(Scalar<O>, Scalar<O>) -> Scalar<O>>,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>
)
    -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
    where T: CommonBounds + Cast<O> + DeviceRepr + CudaType, O: CommonBounds + DeviceRepr + CudaType
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
                &format!("contiguous_{op}_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta
            ).unwrap();
            let size = a.size();
            let compute_cfg = |reduce_size: usize| {
                let mut cfg = compute_kernel_launch_config(a.device(), &reg_info, reduce_size);
                let num_blocks = compute_num_blocks(a.device(), reduce_size, 256, 4);
                let block_dim_x = /*cfg.block_dim.0.next_power_of_two().max(256)*/ 256;
                cfg.block_dim = (block_dim_x, 1, 1);
                // calculate the number of blocks needed, divide by 2 for reduction
                cfg.grid_dim = (num_blocks as u32, 1, 1);
                check_launch_config(a.device(), &cfg).unwrap();
                cfg
            };

            let cfg = compute_cfg(size);
            let mut reduce_size = cfg.grid_dim.0 as usize;
            let tmp_buffer = unsafe { a.device().alloc::<O>(reduce_size).unwrap() };

            (
                unsafe {
                    reduce_kernel.clone().launch(cfg, (&tmp_buffer, a.cuda_slice(), size))
                }
            ).unwrap();

            let reduce_kernel = if has_cumulate {
                a.device()
                    .get_func(&module_name, &format!("contiguous_cumulate_{op}_{}", T::STR))
                    .unwrap()
            } else {
                reduce_kernel
            };

            // keep reducing until the size is 1
            while reduce_size > 1 {
                let cfg = compute_cfg(reduce_size);
                (
                    unsafe {
                        reduce_kernel.clone().launch(cfg, (&tmp_buffer, &tmp_buffer, reduce_size))
                    }
                ).unwrap();
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            let tmp_buffer = tmp_buffer.leak();
            // resize the slice
            let tmp_buffer = unsafe { a.device().upgrade_device_ptr::<O>(tmp_buffer, 1) };
            let mut _res_ptr = unsafe { a.device().upgrade_device_ptr::<O>(res.inner, 1) };
            if let Some(post_op) = &post_op {
                unary_raw_mut(&tmp_buffer, &_res_ptr, post_op).expect("post_op failed");
            } else {
                a.device().dtod_copy(&tmp_buffer, &mut _res_ptr).unwrap();
            }
            _res_ptr.leak();
        },
        |inner_loop_size, _, res, transposed_tensor, new_axes| {
            // move last dim
            let perm = (0..transposed_tensor.ndim()).collect::<Vec<_>>();
            let mut right = perm[perm.len() - new_axes.len()..].to_vec();
            let mut left = perm[..perm.len() - new_axes.len()].to_vec();
            let last = right.pop().unwrap();
            let reduce_size_no_fast_dim = right
                .iter()
                .map(|&x| transposed_tensor.shape()[x] as usize)
                .product::<usize>();
            let fast_dim_size = inner_loop_size;
            right.insert(0, last);
            left.extend(right);
            let transposed_tensor = transposed_tensor.permute(&left).unwrap();

            let kernel_name = if reduce_size_no_fast_dim > 1 {
                format!("contiguous_{op}_fast_dim_include_{}", T::STR)
            } else {
                assert_eq!(reduce_size_no_fast_dim, 1);
                format!("contiguous_{op}_fast_dim_only_{}", T::STR)
            };
            let (reduce_kernel, _) = load_ptx_and_get_data(
                module_name,
                &kernel_name,
                a.device(),
                a.device_cap(),
                &meta
            ).unwrap();

            let compute_cfg = |output_size: usize| {
                let mut cfg = LaunchConfig::for_num_elems(0);
                if reduce_size_no_fast_dim > 1 {
                    cfg.block_dim = (32, 16, 1);
                    cfg.grid_dim = (output_size as u32, 1, 1);
                } else {
                    fn compute_block_dim_x(fast_dim_size: usize) -> u32 {
                        let mut block_dim = 32u32;
                        while block_dim < 512 && block_dim * 2 <= (fast_dim_size as u32) {
                            block_dim *= 2;
                        }
                        block_dim
                    }
                    assert_eq!(reduce_size_no_fast_dim, 1);
                    let block_size = compute_block_dim_x(fast_dim_size);
                    let num_blocks = compute_num_blocks(
                        a.device(),
                        (block_size as usize) * output_size,
                        block_size as usize,
                        4
                    );
                    cfg.block_dim = (block_size, 1, 1);
                    cfg.grid_dim = (num_blocks as u32, 1, 1);
                }
                check_launch_config(a.device(), &cfg).unwrap();
                cfg
            };

            let cfg = compute_cfg(res.size());
            let shape = transposed_tensor.cuda_shape_i32().unwrap();
            let strides = transposed_tensor.cuda_strides_i32().unwrap();
            let fast_divmod = transposed_tensor.cuda_divmod().unwrap();
            let num_elements_per_thread = reduce_size_no_fast_dim.div_ceil(
                cfg.block_dim.1 as usize
            );
            if reduce_size_no_fast_dim > 1 {
                (
                    unsafe {
                        reduce_kernel
                            .clone()
                            .launch(cfg, (
                                res.cuda_slice(),
                                transposed_tensor.cuda_slice(),
                                &fast_divmod,
                                &shape,
                                &strides,
                                transposed_tensor.ndim(),
                                fast_dim_size,
                                num_elements_per_thread,
                                reduce_size_no_fast_dim,
                                new_axes.len() - 1,
                            ))
                    }
                ).unwrap();
            } else {
                (
                    unsafe {
                        reduce_kernel
                            .clone()
                            .launch(cfg, (
                                res.cuda_slice(),
                                transposed_tensor.cuda_slice(),
                                fast_dim_size,
                                res.size(),
                            ))
                    }
                ).unwrap();
            }
        },
        |reduce_size, res, transposed_tensor| {
            let mut cfg = LaunchConfig::for_num_elems(0);
            let reduce_size = reduce_size as u32;
            let block_dim_x = 1024;
            let num_el_per_thread = reduce_size.div_ceil(block_dim_x);
            cfg.block_dim = (block_dim_x as u32, 1, 1);
            cfg.grid_dim = (res.size() as u32, 1, 1);
            check_launch_config(a.device(), &cfg).unwrap();
            let strides = transposed_tensor.cuda_strides_i32().unwrap();
            let shape = transposed_tensor.cuda_divmod().unwrap();
            let kernel_name = format!("contiguous_{op}_fast_dim_no_include_small_{}", T::STR);
            let (reduce_kernel, _) = load_ptx_and_get_data(
                module_name,
                &kernel_name,
                a.device(),
                a.device_cap(),
                &meta
            ).unwrap();
            unsafe {
                reduce_kernel
                    .launch(cfg, (
                        res.cuda_slice(),
                        transposed_tensor.cuda_slice(),
                        &shape,
                        &strides,
                        transposed_tensor.ndim(),
                        reduce_size as usize,
                        res.size(),
                        num_el_per_thread as usize,
                    ))
                    .unwrap();
            }
        },
        |reduce_size, res, transposed_tensor, new_axes| {
            let kernel_name = format!("contiguous_{op}_fast_dim_no_include_{}", T::STR);
            let (reduce_kernel, _) = load_ptx_and_get_data(
                module_name,
                &kernel_name,
                a.device(),
                a.device_cap(),
                &meta
            ).unwrap();

            let compute_cfg = |output_size: usize| {
                let mut cfg = LaunchConfig::for_num_elems(0);
                let fast_dim_size: i64 =
                    transposed_tensor.shape()[transposed_tensor.ndim() - new_axes.len() - 1];
                let max_el_per_thread = 512;
                let min_el_per_thread = 16;
                let mut max_block_size = 512usize;
                let block_x = if fast_dim_size < 32 {
                    last_power_of_two(fast_dim_size)
                } else {
                    32
                };
                max_block_size /= block_x;
                let mut block_y = if reduce_size < max_block_size {
                    last_power_of_two(reduce_size)
                } else {
                    max_block_size
                };
                let total_threads = block_x * block_y;
                assert!(total_threads <= max_block_size);
                let num_el_per_output = a.size() / output_size;
                let curr_num_el_per_thread = num_el_per_output / block_y;
                let adjusted_el_per_thread = curr_num_el_per_thread
                    .min(max_el_per_thread)
                    .max(min_el_per_thread);
                let adjusted_grid_y = (num_el_per_output / adjusted_el_per_thread).min(65536);
                let grid_x = output_size / block_x;
                cfg.block_dim = (block_x, block_y, 1);
                cfg.grid_dim = (output_size.div_ceil(block_x) as u32, adjusted_grid_y, 1);
                check_launch_config(a.device(), &cfg).unwrap();
                cfg
            };

            let cfg = compute_cfg(res.size());
            let shape = transposed_tensor.cuda_divmod().unwrap();
            let strides = transposed_tensor.cuda_strides_i32().unwrap();
            (
                unsafe {
                    reduce_kernel
                        .clone()
                        .launch(cfg, (
                            res.cuda_slice(),
                            transposed_tensor.cuda_slice(),
                            &shape,
                            &strides,
                            transposed_tensor.ndim(),
                            reduce_size,
                            new_axes.len(),
                            res.size(),
                        ))
                }
            ).unwrap();
        }
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

#[track_caller]
pub(crate) fn uncontiguous_reduce<T, O, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    meta: &phf::Map<
        usize,
        (&'static str, &'static phf::Map<&'static str, RegisterInfo>, &'static [&str])
    >,
    module_name: &str,
    op: &str,
    has_cumulate: bool,
    post_op: Option<impl Fn(Scalar<O>, Scalar<O>) -> Scalar<O>>,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>
)
    -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
    where
        T: CommonBounds + Cast<O> + DeviceRepr + CudaType + Cast<f64>,
        O: CommonBounds + DeviceRepr + CudaType
{
    uncontiguos_reduce_template(
        a,
        axes,
        init_val,
        keepdims,
        init_out,
        c,
        |res| {
            let (reduce_kernel, reg_info) = load_ptx_and_get_data(
                module_name,
                &format!("uncontiguous_{op}_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta
            ).unwrap();
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
                cfg.shared_mem_bytes = cfg.block_dim.0 * (std::mem::size_of::<T>() as u32);
                check_launch_config(a.device(), &cfg).unwrap();
                cfg
            };

            let cfg = compute_cfg(size);
            let mut reduce_size = cfg.grid_dim.0 as usize;
            let tmp_buffer = unsafe { a.device().alloc::<O>(reduce_size).unwrap() };
            (
                unsafe {
                    reduce_kernel
                        .clone()
                        .launch(cfg, (
                            &tmp_buffer,
                            a.cuda_slice(),
                            &a.cuda_shape().unwrap(),
                            &a.cuda_strides().unwrap(),
                            a.ndim(),
                            size,
                        ))
                }
            ).unwrap();

            let contiguous_reduce_kernel = a
                .device()
                .get_func(&module_name, &format!("contiguous_{op}_{}", T::STR))
                .unwrap();

            let reduce_kernel = if has_cumulate {
                a.device()
                    .get_func(&module_name, &format!("uncontiguous_cumulate_{op}_{}", T::STR))
                    .unwrap()
            } else {
                contiguous_reduce_kernel
            };

            // keep reducing until the size is 1
            while reduce_size > 1 {
                let cfg = compute_cfg(reduce_size);
                // it is safe to use tmp_buffer as input and output because there is no data racing
                (
                    unsafe {
                        reduce_kernel.clone().launch(cfg, (&tmp_buffer, &tmp_buffer, reduce_size))
                    }
                ).unwrap();
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            let tmp_buffer = tmp_buffer.leak();
            // resize the slice
            let tmp_buffer = unsafe { a.device().upgrade_device_ptr::<O>(tmp_buffer, 1) };
            let mut _res_ptr = unsafe { a.device().upgrade_device_ptr::<O>(res.inner, 1) };
            if let Some(post_op) = &post_op {
                unary_raw_mut(&tmp_buffer, &_res_ptr, post_op).expect("post_op failed");
            } else {
                a.device().dtod_copy(&tmp_buffer, &mut _res_ptr).unwrap();
            }
            _res_ptr.leak();
        },
        |inner_loop_size, inner_loop_size2, res, transposed_tensor| {
            let outer_loop_size = a.size() / (inner_loop_size * inner_loop_size2);
            let (reduce_kernel, reg_info) = load_ptx_and_get_data(
                module_name,
                &format!("contiguous_{op}2_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta
            ).unwrap();
            let mut reduce_size = inner_loop_size * inner_loop_size2;
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
                cfg.shared_mem_bytes = cfg.block_dim.0 * (std::mem::size_of::<T>() as u32);
                check_launch_config(a.device(), &cfg).unwrap();
                cfg
            };

            let cfg = compute_cfg(reduce_size);
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let tmp_buffer = unsafe {
                a.device()
                    .alloc::<O>((cfg.grid_dim.0 as usize) * outer_loop_size)
                    .unwrap()
            };
            for idx in (0..outer_loop_size).step_by(max_grid_dim_y as usize) {
                (
                    unsafe {
                        reduce_kernel
                            .clone()
                            .launch(cfg, (
                                &tmp_buffer,
                                transposed_tensor.cuda_slice(),
                                &shape,
                                &strides,
                                transposed_tensor.ndim(),
                                idx,
                                reduce_size,
                                outer_loop_size,
                                cfg.grid_dim.0 as usize,
                            ))
                    }
                ).unwrap();
            }

            reduce_size = cfg.grid_dim.0 as usize;
            let reduce_kernel = a
                .device()
                .get_func(&module_name, &format!("contiguous_{op}22_{}", O::STR))
                .unwrap();
            while reduce_size > 1 {
                let cfg = compute_cfg(reduce_size);
                for idx in (0..outer_loop_size).step_by(max_grid_dim_y as usize) {
                    (
                        unsafe {
                            reduce_kernel
                                .clone()
                                .launch(cfg, (
                                    &tmp_buffer,
                                    &tmp_buffer,
                                    idx,
                                    reduce_size,
                                    outer_loop_size,
                                    cfg.grid_dim.0 as usize,
                                ))
                        }
                    ).unwrap();
                }
                reduce_size = cfg.grid_dim.0 as usize;
            }
            a.device().synchronize().unwrap();
            let tmp_buffer = tmp_buffer.leak();
            // resize the slice
            let tmp_buffer = unsafe { a.device().upgrade_device_ptr::<O>(tmp_buffer, res.size()) };
            if let Some(post_op) = &post_op {
                unary_raw_mut(&tmp_buffer, &tmp_buffer, post_op).expect("post_op failed");
            }
            strided_copy(&tmp_buffer, &mut res.clone()).expect("strided_copy failed");
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
                &format!("contiguous_{op}3_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta
            ).unwrap();
            let block_dim_x = 32; // x dimension must be 32, as it is the warp size
            let block_dim_y = 32; // y dimension depends on the register used, currently 32
            let mut reduce_size = inner_loop_size_2;
            let grid_dim_x = (a.size() / reduce_size).div_ceil(block_dim_x as usize) as u32;
            let launch_cfg = LaunchConfig {
                block_dim: (block_dim_x, block_dim_y, 1),
                grid_dim: (grid_dim_x, reduce_size.div_ceil(block_dim_y as usize) as u32, 1),
                shared_mem_bytes: block_dim_x * block_dim_y * (std::mem::size_of::<T>() as u32),
            };
            check_launch_config(a.device(), &launch_cfg).unwrap();
            let shape = transposed_tensor.cuda_shape().unwrap();
            let strides = transposed_tensor.cuda_strides().unwrap();
            let mut height = launch_cfg.grid_dim.1;
            let tmp_buffer = unsafe {
                a.device()
                    .alloc::<O>(result.size() * (height as usize))
                    .unwrap()
            };
            (
                unsafe {
                    reduce_kernel.launch(launch_cfg, (
                        &tmp_buffer,
                        transposed_tensor.cuda_slice(),
                        &shape,
                        &strides,
                        transposed_tensor.ndim(),
                        result.size(),
                        reduce_size,
                    ))
                }
            ).unwrap();
            let (reduce_kernel, _) = load_ptx_and_get_data(
                module_name,
                &format!("contiguous_{op}33_{}", T::STR),
                a.device(),
                a.device_cap(),
                &meta
            ).unwrap();
            while height > 1 {
                reduce_size = height as usize;
                let launch_cfg = LaunchConfig {
                    block_dim: (block_dim_x, block_dim_y, 1),
                    grid_dim: (grid_dim_x, reduce_size.div_ceil(block_dim_y as usize) as u32, 1),
                    shared_mem_bytes: block_dim_x * block_dim_y * (std::mem::size_of::<T>() as u32),
                };
                check_launch_config(a.device(), &launch_cfg).unwrap();
                (
                    unsafe {
                        reduce_kernel
                            .clone()
                            .launch(launch_cfg, (
                                &tmp_buffer,
                                &tmp_buffer,
                                transposed_tensor.ndim(),
                                result.size(),
                                reduce_size,
                            ))
                    }
                ).unwrap();
                height = launch_cfg.grid_dim.1;
            }
            a.device().synchronize().unwrap();
            let tmp_buffer = tmp_buffer.leak();
            // resize the slice
            let tmp_buffer = unsafe {
                a.device().upgrade_device_ptr::<O>(tmp_buffer, result.size())
            };
            if let Some(post_op) = &post_op {
                unary_raw_mut(&tmp_buffer, &tmp_buffer, post_op).expect("post_op failed");
            }
            strided_copy(&tmp_buffer, &mut result.clone()).expect("strided_copy failed");
        }
    )
}
