use crate::backend::Cpu;
use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;
use crate::tensor_base::_Tensor;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_iterator::iterator_traits::ParStridedIteratorSimd;
use hpt_iterator::iterator_traits::ParStridedIteratorSimdZip;
use hpt_iterator::TensorIterator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::FloatOutUnary;
use hpt_types::type_promote::NormalOut;

use super::conv2d::conv2d;
use super::microkernel_trait::Conv2dMicroKernel;
use hpt_types::traits::VecTrait;

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
) -> std::result::Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: CommonBounds + Conv2dMicroKernel + MatmulMicroKernel,
    T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
    T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
    i64: Cast<T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    let mut conv_res = conv2d(input, kernels, bias, steps, padding, dilation)?;
    // let eps_vec = T::Vec::splat(eps);
    // if let Some(bias) = bias {
    //     conv_res
    //         .par_iter_mut_simd()
    //         .zip(mean.par_iter_simd())
    //         .zip(var.par_iter_simd())
    //         .zip(gamma.par_iter_simd())
    //         .zip(beta.par_iter_simd())
    //         .zip(bias.par_iter_simd())
    //         .for_each(
    //             |(((((out, mean), var), gamma), beta), bias)| {
    //                 *out = out
    //                     ._sub(mean)
    //                     ._mul(gamma)
    //                     ._div(var._add(eps)._sqrt())
    //                     ._add(beta)
    //                     ._add(bias);
    //             },
    //             |(((((out, mean), var), gamma), beta), bias)| {
    //                 let res = out
    //                     .read_unaligned()
    //                     ._sub(mean)
    //                     ._mul(gamma)
    //                     ._div(var._add(eps_vec)._sqrt())
    //                     ._add(beta)
    //                     ._add(bias);
    //                 out.write_unaligned(res);
    //             },
    //         );
    // } else {
    //     conv_res
    //         .par_iter_mut_simd()
    //         .zip(mean.par_iter_simd())
    //         .zip(var.par_iter_simd())
    //         .zip(gamma.par_iter_simd())
    //         .zip(beta.par_iter_simd())
    //         .for_each(
    //             |((((out, mean), var), gamma), beta)| {
    //                 *out = out
    //                     ._sub(mean)
    //                     ._mul(gamma)
    //                     ._div(var._add(eps)._sqrt())
    //                     ._add(beta);
    //             },
    //             |((((out, mean), var), gamma), beta)| {
    //                 let res = out
    //                     .read_unaligned()
    //                     ._sub(mean)
    //                     ._mul(gamma)
    //                     ._div(var._add(eps_vec)._sqrt())
    //                     ._add(beta);
    //                 out.write_unaligned(res);
    //             },
    //         );
    // }

    Ok(conv_res)
}
