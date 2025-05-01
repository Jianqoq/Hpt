use crate::ops::tensor::matmul::microkernel_trait::MatmulMicroKernel;
use crate::ops::tensor::normalization::batch_norm::batch_norm;
use crate::Tensor;

use super::microkernel_trait::Conv2dMicroKernel;
use hpt_common::error::base::TensorError;
use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::ToDType;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::FloatOutUnary;
use hpt_types::type_promote::NormalOutPromote;

#[track_caller]
pub(crate) fn batchnorm_conv2d<T>(
    input: &Tensor,
    kernels: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    bias: Option<&Tensor>,
    eps: T,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>
)
    -> Result<Tensor, TensorError>
    where
        T: CommonBounds +
            Conv2dMicroKernel +
            MatmulMicroKernel +
            Cast<<T as NormalOutPromote>::Intermediate> + ToDType,
        <T as NormalOutPromote>::Intermediate: CommonBounds + Cast<T>,
        T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
        T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>
{
    let conv_res = input.conv2d(kernels, bias, steps, padding, dilation)?;

    batch_norm(
        &conv_res,
        mean,
        var,
        gamma,
        beta,
        eps,
        post_scalar,
        post_vec,
        Some(conv_res.clone()),
    )?;

    Ok(conv_res)
}
