use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cuda,
};
use hpt_common::error::{base::TensorError, shape::ShapeError};
use hpt_cudakernels::SOFTMAX;
use hpt_traits::{
    ops::creation::TensorCreator,
    tensor::{CommonBounds, TensorInfo},
};
use hpt_types::{
    dtype::CudaType,
    into_scalar::Cast,
    type_promote::{FloatOutBinary, FloatOutUnary, FloatOutUnaryPromote, NormalOut},
};

use crate::{
    backends::{
        common::reduce::rearrange_array,
        cuda::cuda_utils::{check_launch_config, load_ptx_and_get_data},
    },
    tensor_base::_Tensor,
};
use hpt_traits::ops::shape_manipulate::ShapeManipulate;

pub(crate) fn calculate_best_block_size_y_warp(kernel: &cudarc::driver::CudaFunction) -> u32 {
    let block_size_y = [1, 2, 4];
    let mut max_active_blocks = 0;
    let mut best_block_size_y = 0;
    for block_size_y in block_size_y {
        let size = 32 * block_size_y;
        let max = kernel
            .occupancy_max_active_blocks_per_multiprocessor(size, 0, None)
            .expect("occupancy failed");
        if max >= max_active_blocks {
            max_active_blocks = max;
            best_block_size_y = block_size_y;
        }
    }
    best_block_size_y
}

pub(crate) fn normalize_prepare<T: CommonBounds, O: CommonBounds, const DEVICE: usize, A>(
    a: &_Tensor<T, Cuda, DEVICE, A>,
    axis: usize,
    c: Option<_Tensor<O, Cuda, DEVICE, A>>,
) -> std::result::Result<
    (
        _Tensor<T, Cuda, DEVICE, A>,
        _Tensor<O, Cuda, DEVICE, A>,
        Vec<usize>,
    ),
    TensorError,
>
where
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
    T: CudaType + DeviceRepr,
    O: CudaType + DeviceRepr,
{
    // get permute order, we move to_reduce axes to the end
    let mut transposed_axis = rearrange_array(a.ndim(), &[axis]);

    // sort the transposed axis based on the stride, ordering the axis can increase the cpu cache hitting rate when we do iteration
    transposed_axis[..a.ndim() - 1].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));
    transposed_axis[a.ndim() - 1..].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));

    let res = if let Some(out) = c {
        // we need a better logic to verify the out is valid.
        // we need to get the real size and compare the real size with the res_shape
        ShapeError::check_inplace_out_layout_valid(a.shape(), out.layout())?;
        Ok(out)
    } else {
        _Tensor::<O, Cuda, DEVICE, A>::empty(a.shape())
    };
    Ok((
        a.permute(&transposed_axis)?,
        res?.permute(&transposed_axis)?,
        transposed_axis,
    ))
}

#[track_caller]
pub(crate) fn contiguous_softmax<T, O, const DEVICE: usize, A>(
    a: &_Tensor<T, Cuda, DEVICE, A>,
    axis: i64,
    c: Option<_Tensor<O, Cuda, DEVICE, A>>,
    is_log_softmax: bool,
) -> Result<_Tensor<O, Cuda, DEVICE, A>, TensorError>
where
    T: CommonBounds + Cast<O> + FloatOutUnary<Output = O> + CudaType + DeviceRepr,
    O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O> + CudaType + DeviceRepr,
    <T as FloatOutUnaryPromote>::Intermediate: DeviceRepr,
    T::Vec: FloatOutUnary<Output = O::Vec>,
    O::Vec: FloatOutBinary<Output = O::Vec>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    let axis = (if axis < 0 {
        axis + (a.ndim() as i64)
    } else {
        axis
    }) as usize;
    let (transposed_tensor, res, transposed_axis) = normalize_prepare(a, axis, c)?;

    let a_last_stride = transposed_tensor.strides()[a.ndim() - 1];
    let inner_loop_size = transposed_tensor.shape()[a.ndim() - 1] as i32;
    let outer_loop_size = transposed_tensor.shape()[..a.ndim() - 1]
        .iter()
        .product::<i64>();
    let op_name = if is_log_softmax {
        "logsoftmax"
    } else {
        "softmax"
    };
    if inner_loop_size <= 1024 {
        let (kernel, _) = load_ptx_and_get_data(
            op_name,
            &if a_last_stride == 1 {
                format!("{}_{op_name}_warp", T::STR)
            } else {
                format!("{}_{op_name}_warp_uncontiguous", T::STR)
            },
            res.device(),
            res.device_cap(),
            &SOFTMAX,
        )
        .expect("load softmax kernel failed");
        let best_block_size_y = calculate_best_block_size_y_warp(&kernel);
        let cfg = LaunchConfig {
            grid_dim: (
                1,
                ((outer_loop_size as u32 + best_block_size_y - 1) / best_block_size_y)
                    .min(u16::MAX as u32),
                1,
            ),
            block_dim: (32, best_block_size_y, 1),
            shared_mem_bytes: 0,
        };
        check_launch_config(res.device(), &cfg)?;
        if a_last_stride == 1 {
            let inp_slice = a.cuda_slice();
            let out_slice = res.cuda_slice();
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            inp_slice,
                            out_slice,
                            outer_loop_size as i32,
                            inner_loop_size,
                        ),
                    )
                    .expect("launch softmax kernel failed");
            }
        } else {
            let inp_slice = transposed_tensor.cuda_slice();
            let out_slice = res.cuda_slice();
            let divmod = transposed_tensor.cuda_divmod()?;
            let strides = transposed_tensor.cuda_strides_i32()?;
            let ndim = transposed_tensor.ndim() as i32;
            let out_divmod = res.cuda_divmod()?;
            let out_strides = res.cuda_strides_i32()?;
            let out_ndim = res.ndim() as i32;
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            inp_slice,
                            out_slice,
                            outer_loop_size as i32,
                            inner_loop_size,
                            &divmod,
                            &strides,
                            ndim,
                            &out_divmod,
                            &out_strides,
                            out_ndim,
                        ),
                    )
                    .expect("launch softmax kernel failed");
            }
        }
    } else if inner_loop_size <= 1024 * 4 {
        let (kernel, _) = load_ptx_and_get_data(
            op_name,
            &if a_last_stride == 1 {
                format!("{}_{op_name}_block", T::STR)
            } else {
                format!("{}_{op_name}_block_uncontiguous", T::STR)
            },
            res.device(),
            res.device_cap(),
            &SOFTMAX,
        )
        .expect("load softmax kernel failed");
        let cfg = LaunchConfig {
            grid_dim: (1, (outer_loop_size as u32).min(u16::MAX as u32), 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: inner_loop_size as u32
                * std::mem::size_of::<<T as FloatOutUnaryPromote>::Intermediate>() as u32,
        };
        check_launch_config(res.device(), &cfg)?;
        if a_last_stride == 1 {
            let inp_slice = a.cuda_slice();
            let out_slice = res.cuda_slice();
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            inp_slice,
                            out_slice,
                            outer_loop_size as i32,
                            inner_loop_size,
                        ),
                    )
                    .expect("launch softmax kernel failed");
            }
        } else {
            let inp_slice = transposed_tensor.cuda_slice();
            let out_slice = res.cuda_slice();
            let divmod = transposed_tensor.cuda_divmod()?;
            let strides = transposed_tensor.cuda_strides_i32()?;
            let ndim = transposed_tensor.ndim() as i32;
            let out_divmod = res.cuda_divmod()?;
            let out_strides = res.cuda_strides_i32()?;
            let out_ndim = res.ndim() as i32;
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            inp_slice,
                            out_slice,
                            outer_loop_size as i32,
                            inner_loop_size,
                            &divmod,
                            &strides,
                            ndim,
                            &out_divmod,
                            &out_strides,
                            out_ndim,
                        ),
                    )
                    .expect("launch softmax kernel failed");
            }
        }
    } else {
        let (kernel, _) = load_ptx_and_get_data(
            op_name,
            &if a_last_stride == 1 {
                format!("{}_{op_name}_block_large", T::STR)
            } else {
                format!("{}_{op_name}_block_large_uncontiguous", T::STR)
            },
            res.device(),
            res.device_cap(),
            &SOFTMAX,
        )
        .expect("load softmax kernel failed");
        let cfg = LaunchConfig {
            grid_dim: (1, (outer_loop_size as u32).min(u16::MAX as u32), 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        check_launch_config(res.device(), &cfg)?;
        if a_last_stride == 1 {
            let inp_slice = a.cuda_slice();
            let out_slice = res.cuda_slice();
            let buffer = unsafe {
                res.device()
                    .alloc::<<T as FloatOutUnaryPromote>::Intermediate>(
                        inner_loop_size as usize * cfg.grid_dim.1 as usize,
                    )
            }?;
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            inp_slice,
                            out_slice,
                            &buffer,
                            outer_loop_size as i32,
                            inner_loop_size,
                        ),
                    )
                    .expect("launch softmax kernel failed");
            }
        } else {
            let inp_slice = transposed_tensor.cuda_slice();
            let out_slice = res.cuda_slice();
            let buffer = unsafe {
                res.device()
                    .alloc::<<T as FloatOutUnaryPromote>::Intermediate>(
                        inner_loop_size as usize * cfg.grid_dim.1 as usize,
                    )
            }?;
            let divmod = transposed_tensor.cuda_divmod()?;
            let strides = transposed_tensor.cuda_strides_i32()?;
            let ndim = transposed_tensor.ndim() as i32;
            let out_divmod = res.cuda_divmod()?;
            let out_strides = res.cuda_strides_i32()?;
            let out_ndim = res.ndim() as i32;
            unsafe {
                kernel
                    .launch(
                        cfg,
                        (
                            inp_slice,
                            out_slice,
                            &buffer,
                            outer_loop_size as i32,
                            inner_loop_size,
                            &divmod,
                            &strides,
                            ndim,
                            &out_divmod,
                            &out_strides,
                            out_ndim,
                        ),
                    )
                    .expect("launch softmax kernel failed");
            }
        }
    }
    res.permute_inv(&transposed_axis)
}

#[track_caller]
pub(crate) fn uncontiguous_softmax<T, O, const DEVICE: usize, A>(
    a: &_Tensor<T, Cuda, DEVICE, A>,
    axis: i64,
    c: Option<_Tensor<O, Cuda, DEVICE, A>>,
    is_log_softmax: bool,
) -> Result<_Tensor<O, Cuda, DEVICE, A>, TensorError>
where
    T: CommonBounds + Cast<O> + FloatOutUnary<Output = O> + CudaType + DeviceRepr,
    O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O> + CudaType + DeviceRepr,
    T::Vec: FloatOutUnary<Output = O::Vec>,
    O::Vec: FloatOutBinary<Output = O::Vec>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    let axis = (if axis < 0 {
        axis + (a.ndim() as i64)
    } else {
        axis
    }) as usize;
    let (transposed_tensor, res, transposed_axis) = normalize_prepare(a, axis, c)?;

    let a_last_stride = transposed_tensor.strides()[a.ndim() - 1];
    let inner_loop_size = transposed_tensor.shape()[a.ndim() - 1] as i32;
    assert_eq!(a_last_stride, 1);
    let outer_loop_size = transposed_tensor.shape()[..a.ndim() - 1]
        .iter()
        .product::<i64>();
    let op_name = if is_log_softmax {
        "logsoftmax"
    } else {
        "softmax"
    };
    if inner_loop_size <= 1024 {
        let (kernel, _) = load_ptx_and_get_data(
            op_name,
            &format!("{}_{op_name}_warp_uncontiguous", T::STR),
            res.device(),
            res.device_cap(),
            &SOFTMAX,
        )
        .expect("load softmax kernel failed");

        let cfg = LaunchConfig {
            grid_dim: (1, (outer_loop_size as u32).min(u16::MAX as u32), 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: inner_loop_size as u32 * std::mem::size_of::<O>() as u32
                + inner_loop_size as u32 * std::mem::size_of::<T>() as u32,
        };
        check_launch_config(res.device(), &cfg)?;
        let inp_slice = a.cuda_slice();
        let out_slice = res.cuda_slice();
        let divmod = transposed_tensor.cuda_divmod()?;
        let strides = transposed_tensor.cuda_strides_i32()?;
        let ndim = transposed_tensor.ndim();
        unsafe {
            kernel
                .launch(
                    cfg,
                    (
                        inp_slice,
                        out_slice,
                        outer_loop_size as i32,
                        inner_loop_size,
                        &divmod,
                        &strides,
                        ndim as i32,
                    ),
                )
                .expect("launch softmax kernel failed");
        }
    } else if inner_loop_size <= 1024 * 4 {
        let (kernel, _) = load_ptx_and_get_data(
            op_name,
            &format!("{}_{op_name}_block_uncontiguous", T::STR),
            res.device(),
            res.device_cap(),
            &SOFTMAX,
        )
        .expect("load softmax kernel failed");
        let cfg = LaunchConfig {
            grid_dim: (1, (outer_loop_size as u32).min(u16::MAX as u32), 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: inner_loop_size as u32 * std::mem::size_of::<O>() as u32,
        };
        check_launch_config(res.device(), &cfg)?;
        let inp_slice = a.cuda_slice();
        let out_slice = res.cuda_slice();
        let divmod = transposed_tensor.cuda_divmod()?;
        let strides = transposed_tensor.cuda_strides_i32()?;
        let ndim = transposed_tensor.ndim();
        unsafe {
            kernel
                .launch(
                    cfg,
                    (
                        inp_slice,
                        out_slice,
                        outer_loop_size as i32,
                        inner_loop_size,
                        &divmod,
                        &strides,
                        ndim as i32,
                    ),
                )
                .expect("launch softmax kernel failed");
        }
    } else {
        let (kernel, _) = load_ptx_and_get_data(
            op_name,
            &format!("{}_{op_name}_block_large_uncontiguous", T::STR),
            res.device(),
            res.device_cap(),
            &SOFTMAX,
        )
        .expect("load softmax kernel failed");
        let cfg = LaunchConfig {
            grid_dim: (1, (outer_loop_size as u32).min(u16::MAX as u32), 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        check_launch_config(res.device(), &cfg)?;
        let inp_slice = a.cuda_slice();
        let out_slice = res.cuda_slice();
        let buffer = unsafe { res.device().alloc::<T>(inner_loop_size as usize) }?;
        let divmod = transposed_tensor.cuda_divmod()?;
        let strides = transposed_tensor.cuda_strides_i32()?;
        let ndim = transposed_tensor.ndim();
        unsafe {
            kernel
                .launch(
                    cfg,
                    (
                        inp_slice,
                        out_slice,
                        &buffer,
                        outer_loop_size as i32,
                        inner_loop_size,
                        &divmod,
                        &strides,
                        ndim as i32,
                    ),
                )
                .expect("launch softmax kernel failed");
        }
    }
    res.permute_inv(&transposed_axis)
}
