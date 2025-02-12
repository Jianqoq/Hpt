use hpt_traits::{ops::conv::Conv, CommonBounds};
use hpt_types::{into_scalar::Cast, traits::VecTrait, type_promote::NormalOut};

use crate::{
    ops::cpu::kernels::conv2d::{
        conv2d::conv2d, conv2d_group::conv2d_group, conv2d_transpose::conv2d_transpose,
        dwconv2d::dwconv2d,
    },
    tensor_base::_Tensor,
    Cpu,
};

impl<T, const DEVICE: usize> Conv<T> for _Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Cast<T> + NormalOut<Output = T>,
    T::Vec: VecTrait<T> + Copy + Send + Sync + NormalOut<Output = T::Vec>,
    bool: Cast<T>,
{
    type Output = _Tensor<T, Cpu, DEVICE>;

    fn conv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        conv2d(self, kernels, bias, steps, padding, dilation, activation)
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
        conv2d_group(
            self, kernels, bias, steps, padding, dilation, groups, activation,
        )
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
        dwconv2d(self, kernels, bias, steps, padding, dilation, activation)
    }

    fn conv2d_transpose(
        &self,
        kernels: &Self::Output,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        output_padding: [i64; 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        conv2d_transpose(self, kernels, steps, padding, output_padding, dilation)
    }
}
