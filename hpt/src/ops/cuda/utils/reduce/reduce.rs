use std::cmp::Reverse;

use crate::ops::common::reduce::rearrange_array;
use crate::ops::cuda::cuda_slice::CudaSlice;
use crate::ops::cuda::cuda_utils::check_launch_config;
use crate::ops::cuda::cuda_utils::compute_kernel_launch_config;
use crate::ops::cuda::cuda_utils::compute_num_blocks;
use crate::ops::cuda::cuda_utils::load_ptx_and_get_data;
use crate::ops::cuda::utils::launch_cfg::launch_cfg_trait::LaunchConfigUtils;
use crate::ops::cuda::utils::reduce::reduce_template::contiguous_reduce_template;
use crate::tensor_base::_Tensor;
use crate::Cuda;
use crate::TensorCreator;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape::Shape;
use hpt_cudakernels::RegisterInfo;
use hpt_cudakernels::SET_VAL;
use hpt_traits::shape_manipulate::ShapeManipulate;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;

fn compute_reduce_launch_config(
    output_size: usize,
    reduce_size: usize,
    block_x_calc: impl Fn() -> usize,
) -> LaunchConfig {
    let mut cfg = LaunchConfig::for_num_elems(0);
    let max_el_per_thread = 64;
    let min_el_per_thread = 16;
    let mut max_block_size = 512usize;
    let block_x = block_x_calc();
    max_block_size /= block_x;
    let block_y = if reduce_size < max_block_size {
        last_power_of_two(reduce_size)
    } else {
        max_block_size
    };
    let total_threads = block_x * block_y;
    assert!(total_threads <= 512);
    let num_el_per_output = reduce_size;
    let curr_num_el_per_thread = (num_el_per_output / block_y).max(1);
    let adjusted_el_per_thread = curr_num_el_per_thread
        .min(max_el_per_thread)
        .max(min_el_per_thread);
    let grid_y = (curr_num_el_per_thread / adjusted_el_per_thread)
        .min(65536)
        .max(1);
    let grid_x = output_size / block_x;
    cfg.block_dim = (block_x as u32, block_y as u32, 1);
    cfg.grid_dim = (grid_x as u32, grid_y as u32, 1);
    cfg
}

#[track_caller]
pub(crate) fn reduce<T, BufferType, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
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
    op: &str,
    c: Option<_Tensor<T, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<T, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType + Cast<f64>,
    BufferType: DeviceRepr,
{
    if a.is_contiguous() && a.parent().is_none() {
        contiguous_reduce::<T, T, BufferType, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            c,
        )
    } else {
        uncontiguous_reduce::<T, T, BufferType, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            c,
        )
    }
}

#[track_caller]
pub(crate) fn reduce2<T, O, BufferType, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
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
    op: &str,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType + Cast<f64>,
    O: CommonBounds + DeviceRepr + CudaType,
    BufferType: DeviceRepr,
{
    if a.is_contiguous() && a.parent().is_none() {
        contiguous_reduce::<T, O, BufferType, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            c,
        )
    } else {
        uncontiguous_reduce::<T, O, BufferType, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            c,
        )
    }
}

#[track_caller]
pub(crate) fn reduce3<T, O, BufferType, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
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
    op: &str,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType + Cast<f64>,
    O: CommonBounds + DeviceRepr + CudaType,
    BufferType: DeviceRepr,
{
    if a.is_contiguous() && a.parent().is_none() {
        contiguous_reduce::<T, O, BufferType, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            c,
        )
    } else {
        uncontiguous_reduce::<T, O, BufferType, DEVICE_ID>(
            a,
            &axes,
            init_val,
            keepdims,
            init_out,
            meta,
            module_name,
            op,
            c,
        )
    }
}

fn last_power_of_two(val: usize) -> usize {
    if val <= 1 {
        return val;
    }
    if val.count_ones() == 1 {
        return val;
    }

    let highest_bit = usize::BITS - (val - 1).leading_zeros() - 1;
    1 << highest_bit
}

pub(crate) fn fast_all_reduce<T, BufferType, const DEVICE_ID: usize>(
    module_name: &str,
    op: &str,
    inp: &_Tensor<T, Cuda, DEVICE_ID>,
    res: CudaSlice,
    meta: &phf::Map<
        usize,
        (
            &'static str,
            &'static phf::Map<&'static str, RegisterInfo>,
            &'static [&str],
        ),
    >,
) where
    T: CommonBounds + DeviceRepr + CudaType,
    BufferType: DeviceRepr,
{
    let (reduce_kernel, _) = load_ptx_and_get_data(
        module_name,
        &format!("contiguous_{op}_{}", T::STR),
        inp.device(),
        inp.device_cap(),
        &meta,
    )
    .unwrap();
    let size = inp.size();
    let compute_cfg = |reduce_size: usize| {
        let mut cfg = LaunchConfig::for_num_elems(0);
        let block_dim_x = last_power_of_two(reduce_size.next_multiple_of(32)).min(512);
        let num_blocks = compute_num_blocks(inp.device(), reduce_size, block_dim_x, 4);
        cfg.block_dim = (block_dim_x as u32, 1, 1);
        // calculate the number of blocks needed, divide by 2 for reduction
        cfg.grid_dim = (num_blocks as u32, 1, 1);
        check_launch_config(inp.device(), &cfg).unwrap();
        cfg
    };

    let cfg = compute_cfg(size);
    let reduce_size = cfg.grid_dim.0 as usize;
    let tmp_buffer = unsafe { inp.device().alloc::<BufferType>(reduce_size).unwrap() };
    let finished = inp.device().alloc_zeros::<i32>(1).unwrap();
    (unsafe {
        reduce_kernel
            .clone()
            .launch(cfg, (res, &tmp_buffer, inp.cuda_slice(), &finished, size))
    })
    .unwrap();
}

pub(crate) fn slow_all_reduce<T, BufferType, const DEVICE_ID: usize>(
    module_name: &str,
    op: &str,
    inp: &_Tensor<T, Cuda, DEVICE_ID>,
    res: CudaSlice,
    meta: &phf::Map<
        usize,
        (
            &'static str,
            &'static phf::Map<&'static str, RegisterInfo>,
            &'static [&str],
        ),
    >,
) where
    T: CommonBounds + DeviceRepr + CudaType,
    BufferType: DeviceRepr,
{
    let (reduce_kernel, _) = load_ptx_and_get_data(
        module_name,
        &format!("uncontiguous_{op}_{}", T::STR),
        inp.device(),
        inp.device_cap(),
        &meta,
    )
    .unwrap();
    let size = inp.size();
    let compute_cfg = |reduce_size: usize| {
        let mut cfg = LaunchConfig::for_num_elems(0);
        let block_dim_x = last_power_of_two(reduce_size.next_multiple_of(32)).min(512);
        let num_blocks = compute_num_blocks(inp.device(), reduce_size, block_dim_x, 4);
        cfg.block_dim = (block_dim_x as u32, 1, 1);
        // calculate the number of blocks needed, divide by 2 for reduction
        cfg.grid_dim = (num_blocks as u32, 1, 1);
        check_launch_config(inp.device(), &cfg).unwrap();
        cfg
    };

    let cfg = compute_cfg(size);
    let reduce_size = cfg.grid_dim.0 as usize;
    let tmp_buffer = unsafe { inp.device().alloc::<BufferType>(reduce_size).unwrap() };
    let finished = inp.device().alloc_zeros::<i32>(1).unwrap();
    let shape = inp.cuda_divmod().unwrap();
    let strides = inp.cuda_strides_i32().unwrap();
    let ndim = inp.ndim();
    (unsafe {
        reduce_kernel.clone().launch(
            cfg,
            (
                res,
                &tmp_buffer,
                inp.cuda_slice(),
                &finished,
                &shape,
                &strides,
                ndim,
                size,
            ),
        )
    })
    .unwrap();
}

pub(crate) fn not_keep_last_dim<T, O, BufferType, const DEVICE_ID: usize>(
    inner_loop_size: usize,
    res: &_Tensor<O, Cuda, DEVICE_ID>,
    transposed_tensor: &_Tensor<T, Cuda, DEVICE_ID>,
    new_axes: &[usize],
    module_name: &str,
    op: &str,
    meta: &phf::Map<
        usize,
        (
            &'static str,
            &'static phf::Map<&'static str, RegisterInfo>,
            &'static [&str],
        ),
    >,
    is_contiguous: bool,
) where
    T: CommonBounds + DeviceRepr + CudaType,
    O: CommonBounds + DeviceRepr + CudaType,
    BufferType: DeviceRepr,
{
    let perm = (0..transposed_tensor.ndim()).collect::<Vec<_>>();
    let mut right = perm[perm.len() - new_axes.len()..].to_vec();
    let mut left = perm[..perm.len() - new_axes.len()].to_vec();
    let last = right.pop().unwrap();

    let reduce_size_no_fast_dim = right
        .iter()
        .map(|&x| transposed_tensor.shape()[x] as usize)
        .product::<usize>();
    let reduce_size_no_fast_ndim = right.len();
    let fast_dim_size = inner_loop_size;
    right.insert(0, last);
    left.extend(right);
    let transposed_tensor = transposed_tensor.permute(&left).unwrap();

    let kernel_name = if reduce_size_no_fast_ndim >= 1 {
        format!("{op}_fast_dim_include_{}", T::STR)
    } else {
        assert_eq!(reduce_size_no_fast_ndim, 0);
        if is_contiguous {
            format!("contiguous_{op}_fast_dim_only_{}", T::STR)
        } else {
            format!("uncontiguous_{op}_fast_dim_only_{}", T::STR)
        }
    };
    let (reduce_kernel, _) = load_ptx_and_get_data(
        module_name,
        &kernel_name,
        res.device(),
        res.device_cap(),
        &meta,
    )
    .unwrap();

    let compute_cfg = |output_size: usize, fast_dim_size: usize, reduce_size_no_fast_dim: usize| {
        if reduce_size_no_fast_ndim >= 1 {
            let mut cfg = compute_reduce_launch_config(output_size, reduce_size_no_fast_dim, || 32);
            cfg.grid_dim.0 = output_size as u32;
            (cfg, 0)
        } else {
            let mut cfg = LaunchConfig::for_num_elems(0);
            let max_el_per_thread = 256;
            let min_el_per_thread = 16;
            let block_x = if fast_dim_size < 32 {
                last_power_of_two(fast_dim_size as usize)
            } else {
                ((fast_dim_size / 32) * 32).clamp(32, 512)
            };
            let num_el_per_output = fast_dim_size;
            let curr_num_el_per_thread = (num_el_per_output / block_x).max(1);
            if curr_num_el_per_thread > max_el_per_thread {
                let adjusted_el_per_thread = curr_num_el_per_thread
                    .min(max_el_per_thread)
                    .max(min_el_per_thread);
                let grid_x = (curr_num_el_per_thread / adjusted_el_per_thread)
                    .min(i32::MAX as usize)
                    .max(1);
                cfg.block_dim = (block_x as u32, 1, 1);
                cfg.grid_dim = (grid_x as u32, (output_size as u32).min(u16::MAX as u32), 1);
                (cfg, 1)
            } else {
                let num_blocks =
                    compute_num_blocks(res.device(), (block_x) * output_size, block_x, 4);
                cfg.block_dim = (block_x as u32, 1, 1);
                cfg.grid_dim = ((num_blocks as u32).min(i32::MAX as u32), 1, 1);
                (cfg, 2)
            }
        }
    };

    let (mut cfg, case) = compute_cfg(res.size(), fast_dim_size, reduce_size_no_fast_dim);
    cfg.block_dim.0 = last_power_of_two(cfg.block_dim.0.next_multiple_of(32) as usize) as u32;
    check_launch_config(res.device(), &cfg).unwrap();
    match case {
        0 => {
            let strides = transposed_tensor.cuda_strides_i32().unwrap();
            let fast_divmod = transposed_tensor.cuda_divmod().unwrap();
            let tmp_buffer = unsafe {
                res.device()
                    .alloc::<BufferType>(res.size() * cfg.grid_dim.1 as usize)
                    .unwrap()
            };
            let finished = res.device().alloc_zeros::<i32>(res.size()).unwrap();
            (unsafe {
                reduce_kernel.clone().launch(
                    cfg,
                    (
                        res.cuda_slice(),
                        transposed_tensor.cuda_slice(),
                        &tmp_buffer,
                        &finished,
                        &fast_divmod,
                        &strides,
                        transposed_tensor.ndim(),
                        fast_dim_size,
                        reduce_size_no_fast_dim,
                    ),
                )
            })
            .unwrap();
        }
        1 => {
            let tmp_buffer = unsafe {
                res.device()
                    .alloc::<BufferType>(res.size() * cfg.grid_dim.0 as usize)
                    .unwrap()
            };
            let finished = res.device().alloc_zeros::<i32>(res.size()).unwrap();
            if is_contiguous {
                (unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            res.cuda_slice(),
                            transposed_tensor.cuda_slice(),
                            &tmp_buffer,
                            &finished,
                            fast_dim_size,
                            res.size(),
                        ),
                    )
                })
                .unwrap();
            } else {
                let shape = transposed_tensor.cuda_divmod().unwrap();
                let strides = transposed_tensor.cuda_strides_i32().unwrap();
                (unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            res.cuda_slice(),
                            transposed_tensor.cuda_slice(),
                            &tmp_buffer,
                            &finished,
                            &shape,
                            &strides,
                            transposed_tensor.ndim(),
                            fast_dim_size,
                            res.size(),
                            transposed_tensor.strides()[0],
                        ),
                    )
                })
                .unwrap();
            }
        }
        2 => {
            if is_contiguous {
                let kernel_name = format!("contiguous_{op}_small_fast_dim_only_{}", T::STR);
                let (reduce_kernel, _) = load_ptx_and_get_data(
                    module_name,
                    &kernel_name,
                    res.device(),
                    res.device_cap(),
                    &meta,
                )
                .unwrap();
                (unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            res.cuda_slice(),
                            transposed_tensor.cuda_slice(),
                            fast_dim_size,
                            res.size(),
                        ),
                    )
                })
                .unwrap();
            } else {
                let kernel_name = format!("uncontiguous_{op}_small_fast_dim_only_{}", T::STR);
                let (reduce_kernel, _) = load_ptx_and_get_data(
                    module_name,
                    &kernel_name,
                    res.device(),
                    res.device_cap(),
                    &meta,
                )
                .unwrap();
                let shape = transposed_tensor.cuda_divmod().unwrap();
                let strides = transposed_tensor.cuda_strides_i32().unwrap();
                (unsafe {
                    reduce_kernel.clone().launch(
                        cfg,
                        (
                            res.cuda_slice(),
                            transposed_tensor.cuda_slice(),
                            &shape,
                            &strides,
                            transposed_tensor.ndim(),
                            fast_dim_size,
                            res.size(),
                            *transposed_tensor.strides().last().unwrap(),
                        ),
                    )
                })
                .unwrap();
            };
        }
        _ => unreachable!(),
    }
}

pub(crate) fn keep_last_dim<T, O, BufferType, const DEVICE_ID: usize>(
    res: &_Tensor<O, Cuda, DEVICE_ID>,
    transposed_tensor: &_Tensor<T, Cuda, DEVICE_ID>,
    reduce_size: usize,
    new_axes: &[usize],
    module_name: &str,
    op: &str,
    meta: &phf::Map<
        usize,
        (
            &'static str,
            &'static phf::Map<&'static str, RegisterInfo>,
            &'static [&str],
        ),
    >,
) where
    T: CommonBounds + DeviceRepr + CudaType,
    O: CommonBounds + DeviceRepr + CudaType,
    BufferType: DeviceRepr,
{
    let kernel_name = format!("{op}_fast_dim_no_include_{}", T::STR);
    let (reduce_kernel, _) = load_ptx_and_get_data(
        module_name,
        &kernel_name,
        res.device(),
        res.device_cap(),
        &meta,
    )
    .unwrap();
    let fast_dim_size = transposed_tensor.shape()[transposed_tensor.ndim() - new_axes.len() - 1];

    let mut cfg = compute_reduce_launch_config(res.size(), reduce_size, || {
        if fast_dim_size < 32 {
            last_power_of_two(fast_dim_size as usize)
        } else {
            32
        }
    });
    cfg.shared_mem_bytes = cfg.block_size() as u32 * (std::mem::size_of::<BufferType>() as u32);
    let buffer = unsafe {
        res.device()
            .alloc::<BufferType>(res.size() * cfg.grid_dim.1 as usize)
            .unwrap()
    };
    let finished = res.device().alloc_zeros::<i32>(res.size()).unwrap();
    let shape = transposed_tensor.cuda_divmod().unwrap();
    let strides = transposed_tensor.cuda_strides_i32().unwrap();
    (unsafe {
        reduce_kernel.clone().launch(
            cfg,
            (
                res.cuda_slice(),
                &buffer,
                transposed_tensor.cuda_slice(),
                &finished,
                &shape,
                &strides,
                transposed_tensor.ndim(),
                reduce_size,
                res.size(),
            ),
        )
    })
    .unwrap();
}

#[track_caller]
pub(crate) fn contiguous_reduce<T, O, BufferType, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
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
    op: &str,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType,
    O: CommonBounds + DeviceRepr + CudaType,
    BufferType: DeviceRepr,
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
        |res| fast_all_reduce::<T, O, DEVICE_ID>(module_name, op, &a, res, &meta),
        |inner_loop_size, _, res, transposed_tensor, new_axes| {
            not_keep_last_dim::<T, O, BufferType, DEVICE_ID>(
                inner_loop_size,
                res,
                transposed_tensor,
                new_axes,
                module_name,
                op,
                &meta,
                true,
            )
        },
        |reduce_size, res, transposed_tensor, new_axes| {
            keep_last_dim::<T, O, BufferType, DEVICE_ID>(
                res,
                transposed_tensor,
                reduce_size,
                new_axes,
                module_name,
                op,
                &meta,
            )
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

#[track_caller]
pub(crate) fn uncontiguous_reduce<T, O, BufferType, const DEVICE_ID: usize>(
    a: &_Tensor<T, Cuda, DEVICE_ID>,
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
    op: &str,
    c: Option<_Tensor<O, Cuda, DEVICE_ID>>,
) -> std::result::Result<_Tensor<O, Cuda, DEVICE_ID>, TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType + Cast<f64>,
    O: CommonBounds + DeviceRepr + CudaType,
    BufferType: DeviceRepr,
{
    let strides = a.layout.strides().to_vec();
    let shape = a.shape().to_vec();
    let reduce_mask = (0..a.ndim()).map(|i| axes.contains(&i)).collect::<Vec<_>>();
    let mut pair = strides
        .into_iter()
        .zip(reduce_mask.into_iter())
        .zip(shape.into_iter())
        .collect::<Vec<_>>();
    pair.sort_by_key(|((stride, _), _)| Reverse(*stride));

    let last_dim_include = axes.contains(&(a.ndim() - 1));

    let all_reduce = axes.len() == a.ndim();

    let all_reduce_strides =
        if let Ok(new_layout) = a.layout.inplace_reshape(&Shape::new(&[a.size()])) {
            new_layout.strides()[0]
        } else {
            2
        };
    let transposed_axis = rearrange_array(a.layout.ndim(), axes);
    let transposed_tensor = a.permute(transposed_axis)?;
    let res_layout = a.layout.reduce(axes, false)?;
    let res = if let Some(out) = c {
        ShapeError::check_inplace_out_layout_valid(res_layout.shape(), &out.layout())?;
        if init_out {
            let (kernel, reg_info) = load_ptx_and_get_data(
                "set_val",
                &format!("set_val_{}", O::STR),
                a.device(),
                a.device_cap(),
                &SET_VAL,
            )
            .unwrap();
            let cfg =
                compute_kernel_launch_config(out.device(), &reg_info, res_layout.size() as usize);
            unsafe {
                kernel.launch(
                    cfg,
                    (out.cuda_slice(), init_val, res_layout.size() as usize),
                )?
            };
        }
        out.reshape(res_layout.shape())?
    } else {
        let res = _Tensor::<O, Cuda, DEVICE_ID>::empty(res_layout.shape())?;
        res
    };
    if all_reduce {
        if all_reduce_strides == 1 {
            fast_all_reduce::<T, BufferType, DEVICE_ID>(
                module_name,
                op,
                a,
                res.cuda_slice(),
                &meta,
            );
        } else {
            slow_all_reduce::<T, BufferType, DEVICE_ID>(
                module_name,
                op,
                a,
                res.cuda_slice(),
                &meta,
            );
        }
    } else {
        if last_dim_include {
            not_keep_last_dim::<T, O, BufferType, DEVICE_ID>(
                *a.shape().last().unwrap() as usize,
                &res,
                &transposed_tensor,
                &axes,
                module_name,
                op,
                &meta,
                false,
            )
        } else {
            let reduce_size = a.size() / res.size();
            keep_last_dim::<T, O, BufferType, DEVICE_ID>(
                &res,
                &transposed_tensor,
                reduce_size,
                &axes,
                module_name,
                op,
                &meta,
            )
        }
    }

    res.reshape(a.layout.reduce(axes, keepdims)?.shape())
}
