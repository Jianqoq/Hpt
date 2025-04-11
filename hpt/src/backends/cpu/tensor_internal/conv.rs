use hpt_traits::{ ops::conv::{ Conv, ConvBatchNorm }, tensor::CommonBounds };
use hpt_types::{
    into_scalar::Cast,
    type_promote::{ FloatOutBinary, FloatOutUnary, NormalOutPromote },
};

use crate::{
    backends::cpu::kernels::conv2d::{
        batchnorm_conv2d::batchnorm_conv2d,
        conv2d_group::conv2d_group,
        conv2d,
        conv2d_new_mp,
        conv2d_transpose::conv2d_transpose,
        dwconv2d::dwconv2d,
        microkernel_trait::Conv2dMicroKernel,
    },
    tensor_base::_Tensor,
};
use hpt_allocator::{ traits::{ Allocator, AllocatorOutputRetrive }, Cpu };

impl<T, const DEVICE: usize, Al> Conv<T>
    for _Tensor<T, Cpu, DEVICE, Al>
    where
        T: CommonBounds + Conv2dMicroKernel + Cast<<T as NormalOutPromote>::Intermediate>,
        <T as NormalOutPromote>::Intermediate: CommonBounds + Cast<T>,
        bool: Cast<T>,
        Al: Allocator + Send + Sync,
        Al::Output: AllocatorOutputRetrive
{
    type Output = _Tensor<T, Cpu, DEVICE, Al>;

    fn conv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2]
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        if T::STR == "bf16" {
            conv2d_new_mp::conv2d(self, kernels, bias, steps, padding, dilation)
        } else if T::STR == "f16" && !cfg!(target_feature = "neon") {
            conv2d_new_mp::conv2d(self, kernels, bias, steps, padding, dilation)
        } else {
            conv2d::conv2d(self, kernels, bias, steps, padding, dilation)
        }
    }

    fn conv2d_group(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        groups: i64,
        activation: Option<fn(<T>::Vec) -> <T>::Vec>
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        conv2d_group(self, kernels, bias, steps, padding, dilation, groups, activation)
    }

    fn dwconv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(<T>::Vec) -> <T>::Vec>
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        dwconv2d(self, kernels, bias, steps, padding, dilation, activation)
    }

    fn conv2d_transpose(
        &self,
        kernels: &Self::Output,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        output_padding: [i64; 2],
        dilation: [i64; 2]
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        conv2d_transpose(self, kernels, steps, padding, output_padding, dilation)
    }
}

impl<T, const DEVICE: usize, A> ConvBatchNorm<T>
    for _Tensor<T, Cpu, DEVICE, A>
    where
        T: CommonBounds + Conv2dMicroKernel,
        T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
        T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
        bool: Cast<T>,
        A: Allocator + Send + Sync,
        A::Output: AllocatorOutputRetrive
{
    type Output = _Tensor<T, Cpu, DEVICE, A>;
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
        activation: Option<fn(<T>::Vec) -> <T>::Vec>
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        batchnorm_conv2d(
            self,
            kernels,
            mean,
            var,
            gamma,
            beta,
            bias,
            eps,
            steps,
            padding,
            dilation,
        )
    }
}
