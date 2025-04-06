use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cuda,
};
use hpt_common::error::{base::TensorError, shape::ShapeError};
use hpt_cudakernels::{LAYERNORM, LAYERNORM_POST};
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
    backends::cuda::cuda_utils::{
        check_launch_config, compute_kernel_launch_config, load_ptx_and_get_data,
    },
    tensor_base::_Tensor,
};
use hpt_traits::ops::shape_manipulate::ShapeManipulate;

pub(crate) fn calculate_best_block_size_y(
    kernel: &cudarc::driver::CudaFunction,
    warp_size: u32,
) -> u32 {
    let block_size_y = [1, 2, 4];
    let mut max_active_blocks = 0;
    let mut best_block_size_y = 0;
    for block_size_y in block_size_y {
        let size = warp_size * block_size_y;
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
pub(crate) fn layernorm<T, O, const DEVICE: usize, A>(
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
        let best_block_size_y = calculate_best_block_size_y(&kernel, 32);
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
                .expect("launch layernorm kernel failed");
        }
    } else if inner_loop_size <= 1024 * 4 {
        let (kernel, _) = load_ptx_and_get_data(
            "layernorm",
            &format!("{}_layernorm_block", T::STR),
            res.device(),
            res.device_cap(),
            &LAYERNORM,
        )
        .expect("load layernorm kernel failed");
        let best_block_size_y = calculate_best_block_size_y(&kernel, 128);
        let cfg = LaunchConfig {
            grid_dim: (
                1,
                ((outer_loop_size as u32 + best_block_size_y - 1) / best_block_size_y)
                    .min(u16::MAX as u32),
                1,
            ),
            block_dim: (128, best_block_size_y, 1),
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
                .expect("launch layernorm kernel failed");
        }
    } else {
        let (kernel, _) = load_ptx_and_get_data(
            "layernorm",
            &format!("{}_layernorm_block_large", T::STR),
            res.device(),
            res.device_cap(),
            &LAYERNORM,
        )
        .expect("load layernorm kernel failed");
        let cfg = LaunchConfig {
            grid_dim: (1, (outer_loop_size as u32).min(u16::MAX as u32), 1),
            block_dim: (1024, 1, 1),
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
                .expect("launch layernorm kernel failed");
        }
    }
    match (gamma, beta) {
        (None, None) => Ok(res),
        (None, Some(beta)) => hpt_traits::ops::binary::NormalBinOps::add_(&res, beta, res.clone()),
        (Some(gamma), None) => {
            hpt_traits::ops::binary::NormalBinOps::mul_(&res, gamma, res.clone())
        }
        (Some(gamma), Some(beta)) => {
            let (kernel, reg_info) = load_ptx_and_get_data(
                "layernorm_post",
                &format!("layernorm_post_{}", O::STR),
                res.device(),
                res.device_cap(),
                &LAYERNORM_POST,
            )?;
            let cfg = compute_kernel_launch_config(res.device(), &reg_info, res.size());
            let in_out = res.cuda_slice();
            let gamma_slice = gamma.cuda_slice();
            let beta_slice = beta.cuda_slice();
            let size = res.size();
            let channels = normalize_size as usize;
            unsafe { kernel.launch(cfg, (in_out, gamma_slice, beta_slice, size, channels)) }?;
            Ok(res)
        }
    }
}
