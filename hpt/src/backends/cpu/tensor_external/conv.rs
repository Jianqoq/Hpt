use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_traits::ops::conv::ConvBatchNorm;
use hpt_traits::{ops::conv::Conv, tensor::CommonBounds};
use hpt_types::type_promote::NormalOutPromote;
use hpt_types::{
    into_scalar::Cast,
    traits::VecTrait,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};

use crate::Tensor;
use crate::backends::cpu::kernels::conv2d::microkernel_trait::Conv2dMicroKernel;

impl<T, const DEVICE: usize> Conv<T> for Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Cast<T> + NormalOut<Output = T> + Conv2dMicroKernel + Cast<<T as NormalOutPromote>::Intermediate>,
    <T as NormalOutPromote>::Intermediate: CommonBounds + Cast<T>,
    T::Vec: VecTrait<T> + Copy + Send + Sync + NormalOut<Output = T::Vec>,
    bool: Cast<T>,
{
    type Output = Tensor<T, Cpu, DEVICE>;

    fn conv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        Ok(self
            .inner
            .conv2d(
                kernels.inner.as_ref(),
                bias.map(|b| b.inner.as_ref()),
                steps,
                padding,
                dilation,
            )?
            .into())
    }

    fn conv2d_group(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        groups: i64,
        activation: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        Ok(self
            .inner
            .conv2d_group(
                kernels.inner.as_ref(),
                bias.map(|b| b.inner.as_ref()),
                steps,
                padding,
                dilation,
                groups,
                activation,
            )?
            .into())
    }

    fn dwconv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        Ok(self
            .inner
            .dwconv2d(
                kernels.inner.as_ref(),
                bias.map(|b| b.inner.as_ref()),
                steps,
                padding,
                dilation,
                activation,
            )?
            .into())
    }

    fn conv2d_transpose(
        &self,
        kernels: &Self::Output,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        output_padding: [i64; 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        Ok(self
            .inner
            .conv2d_transpose(
                kernels.inner.as_ref(),
                steps,
                padding,
                output_padding,
                dilation,
            )?
            .into())
    }
}

impl<T, const DEVICE: usize, A> ConvBatchNorm<T> for Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds,
    T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
    T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
    bool: Cast<T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cpu, DEVICE, A>;
    fn batchnorm_conv2d(
        &self,
        kernels: &Self::Output,
        mean: &Self::Output,
        var: &Self::Output,
        gamma: &Self::Output,
        beta: &Self::Output,
        bias: Option<&Self::Output>,
        eps: T,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        Ok(self
            .inner
            .batchnorm_conv2d(
                kernels.inner.as_ref(),
                mean.inner.as_ref(),
                var.inner.as_ref(),
                gamma.inner.as_ref(),
                beta.inner.as_ref(),
                bias.map(|b| b.inner.as_ref()),
                eps,
                steps,
                padding,
                dilation,
                activation,
            )?
            .into())
    }
}
