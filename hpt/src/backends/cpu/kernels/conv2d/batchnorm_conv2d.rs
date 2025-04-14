use crate::backend::Cpu;
use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;
use crate::backends::cpu::kernels::normalization::batch_norm::batch_norm;
use crate::tensor_base::_Tensor;
use hpt_allocator::traits::{ Allocator, AllocatorOutputRetrive };
use hpt_common::error::base::TensorError;
use hpt_traits::tensor::CommonBounds;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::FloatOutUnary;

use super::conv2d::conv2d;
use super::microkernel_trait::Conv2dMicroKernel;

#[track_caller]
pub(crate) fn batchnorm_conv2d<T, const DEVICE: usize, A>(
    input: &_Tensor<T, Cpu, DEVICE, A>,
    kernels: &_Tensor<T, Cpu, DEVICE, A>,
    mean: &_Tensor<T, Cpu, DEVICE, A>,
    var: &_Tensor<T, Cpu, DEVICE, A>,
    gamma: &_Tensor<T, Cpu, DEVICE, A>,
    beta: &_Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    eps: T,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>
)
    -> std::result::Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
    where
        T: CommonBounds + Conv2dMicroKernel + MatmulMicroKernel,
        T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
        T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
        A: Allocator + Send + Sync,
        A::Output: AllocatorOutputRetrive
{
    let conv_res = conv2d(input, kernels, bias, steps, padding, dilation, None, None)?;

    batch_norm(
        &conv_res,
        mean,
        var,
        gamma,
        beta,
        eps,
        post_scalar,
        post_vec,
        Some(conv_res.clone())
    )?;

    Ok(conv_res)
}
