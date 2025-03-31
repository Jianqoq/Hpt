use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cuda,
};
use hpt_common::error::{base::TensorError, shape::ShapeError};
use hpt_cudakernels::{LAYERNORM, SOFTMAX};
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

#[track_caller]
pub(crate) fn contiguous_layernorm<T, O, const DEVICE: usize, A>(
    a: &_Tensor<T, Cuda, DEVICE, A>,
    gamma: Option<&_Tensor<O, Cuda, DEVICE, A>>,
    beta: Option<&_Tensor<O, Cuda, DEVICE, A>>,
    eps: O,
    normalized_shape: &[i64],
    c: Option<_Tensor<O, Cuda, DEVICE, A>>,
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
    let normalize_size = normalized_shape.iter().product::<i64>();
    let not_normalize_size = a.size() as i64 / normalize_size;

    let inp = a.reshape(&[not_normalize_size, normalize_size])?;
    let res = if let Some(out) = c {
        // we need a better logic to verify the out is valid.
        // we need to get the real size and compare the real size with the res_shape
        ShapeError::check_inplace_out_layout_valid(a.shape(), out.layout())?;
        Ok(out)
    } else {
        _Tensor::<O, Cuda, DEVICE, A>::empty(a.shape())
    }?;

    let a_last_stride = inp.strides()[inp.ndim() - 1];
    assert!(a_last_stride == 1);
    let inner_loop_size = inp.shape()[inp.ndim() - 1] as i32;
    let outer_loop_size = inp.shape()[..inp.ndim() - 1].iter().product::<i64>();
    if inner_loop_size <= 1024 {
        let (kernel, _) = load_ptx_and_get_data(
            "layernorm",
            &format!("{}_layernorm_warp", T::STR),
            res.device(),
            res.device_cap(),
            &LAYERNORM,
        )
        .expect("load layernorm kernel failed");
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
        let inp_slice = a.cuda_slice();
        let out_slice = res.cuda_slice();
        unsafe {
            kernel
                .launch(
                    cfg,
                    (
                        inp_slice,
                        out_slice,
                        eps,
                        outer_loop_size as i32,
                        inner_loop_size,
                    ),
                )
                .expect("launch softmax kernel failed");
        }
    } else if inner_loop_size <= 1024 * 4 {
        // let (kernel, _) = load_ptx_and_get_data(
        //     "softmax",
        //     &if a_last_stride == 1 {
        //         format!("{}_softmax_block", T::STR)
        //     } else {
        //         format!("{}_softmax_block_uncontiguous", T::STR)
        //     },
        //     res.device(),
        //     res.device_cap(),
        //     &SOFTMAX,
        // )
        // .expect("load softmax kernel failed");
        // let cfg = LaunchConfig {
        //     grid_dim: (1, (outer_loop_size as u32).min(u16::MAX as u32), 1),
        //     block_dim: (1024, 1, 1),
        //     shared_mem_bytes: inner_loop_size as u32
        //         * std::mem::size_of::<<T as FloatOutUnaryPromote>::Intermediate>() as u32,
        // };
        // check_launch_config(res.device(), &cfg)?;
        // if a_last_stride == 1 {
        //     let inp_slice = a.cuda_slice();
        //     let out_slice = res.cuda_slice();
        //     unsafe {
        //         kernel
        //             .launch(
        //                 cfg,
        //                 (
        //                     inp_slice,
        //                     out_slice,
        //                     outer_loop_size as i32,
        //                     inner_loop_size,
        //                 ),
        //             )
        //             .expect("launch softmax kernel failed");
        //     }
        // } else {
        //     let inp_slice = transposed_tensor.cuda_slice();
        //     let out_slice = res.cuda_slice();
        //     let divmod = transposed_tensor.cuda_divmod()?;
        //     let strides = transposed_tensor.cuda_strides_i32()?;
        //     let ndim = transposed_tensor.ndim() as i32;
        //     let out_divmod = res.cuda_divmod()?;
        //     let out_strides = res.cuda_strides_i32()?;
        //     let out_ndim = res.ndim() as i32;
        //     unsafe {
        //         kernel
        //             .launch(
        //                 cfg,
        //                 (
        //                     inp_slice,
        //                     out_slice,
        //                     outer_loop_size as i32,
        //                     inner_loop_size,
        //                     &divmod,
        //                     &strides,
        //                     ndim,
        //                     &out_divmod,
        //                     &out_strides,
        //                     out_ndim,
        //                 ),
        //             )
        //             .expect("launch softmax kernel failed");
        //     }
        // }
    } else {
        // let (kernel, _) = load_ptx_and_get_data(
        //     "softmax",
        //     &if a_last_stride == 1 {
        //         format!("{}_softmax_block_large", T::STR)
        //     } else {
        //         format!("{}_softmax_block_large_uncontiguous", T::STR)
        //     },
        //     res.device(),
        //     res.device_cap(),
        //     &SOFTMAX,
        // )
        // .expect("load softmax kernel failed");
        // let cfg = LaunchConfig {
        //     grid_dim: (1, (outer_loop_size as u32).min(u16::MAX as u32), 1),
        //     block_dim: (1024, 1, 1),
        //     shared_mem_bytes: 0,
        // };
        // check_launch_config(res.device(), &cfg)?;
        // if a_last_stride == 1 {
        //     let inp_slice = a.cuda_slice();
        //     let out_slice = res.cuda_slice();
        //     let buffer = unsafe {
        //         res.device()
        //             .alloc::<<T as FloatOutUnaryPromote>::Intermediate>(
        //                 inner_loop_size as usize * cfg.grid_dim.1 as usize,
        //             )
        //     }?;
        //     unsafe {
        //         kernel
        //             .launch(
        //                 cfg,
        //                 (
        //                     inp_slice,
        //                     out_slice,
        //                     &buffer,
        //                     outer_loop_size as i32,
        //                     inner_loop_size,
        //                 ),
        //             )
        //             .expect("launch softmax kernel failed");
        //     }
        // } else {
        //     let inp_slice = transposed_tensor.cuda_slice();
        //     let out_slice = res.cuda_slice();
        //     let buffer = unsafe {
        //         res.device()
        //             .alloc::<<T as FloatOutUnaryPromote>::Intermediate>(
        //                 inner_loop_size as usize * cfg.grid_dim.1 as usize,
        //             )
        //     }?;
        //     let divmod = transposed_tensor.cuda_divmod()?;
        //     let strides = transposed_tensor.cuda_strides_i32()?;
        //     let ndim = transposed_tensor.ndim() as i32;
        //     let out_divmod = res.cuda_divmod()?;
        //     let out_strides = res.cuda_strides_i32()?;
        //     let out_ndim = res.ndim() as i32;
        //     unsafe {
        //         kernel
        //             .launch(
        //                 cfg,
        //                 (
        //                     inp_slice,
        //                     out_slice,
        //                     &buffer,
        //                     outer_loop_size as i32,
        //                     inner_loop_size,
        //                     &divmod,
        //                     &strides,
        //                     ndim,
        //                     &out_divmod,
        //                     &out_strides,
        //                     out_ndim,
        //                 ),
        //             )
        //             .expect("launch softmax kernel failed");
        //     }
        // }
    }
    Ok(res)
}
