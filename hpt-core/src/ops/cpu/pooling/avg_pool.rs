use crate::ops::cpu::pooling::common::pooling_template;
use crate::tensor_base::_Tensor;
use crate::Cpu;
use crate::Tensor;
use hpt_common::error::base::TensorError;
use hpt_common::shape::shape::Shape;
use hpt_traits::ops::pooling::FloatOutPooling;
use hpt_traits::CommonBounds;
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::NormalOut;
use hpt_types::vectors::traits::*;

use super::common::adaptive_pooling_template;

impl<T, const DEVICE: usize> FloatOutPooling for _Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds
        + Cast<T>
        + NormalOut<Output = T>
        + FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
    <T as FloatOutBinary>::Output:
        CommonBounds + FloatOutBinary<Output = <T as FloatOutBinary>::Output>,
    T::Vec: VecTrait<T>
        + Copy
        + Send
        + Sync
        + NormalOut<Output = T::Vec>
        + FloatOutBinary<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
    bool: Cast<T>,
    i64: Cast<<T as FloatOutBinary>::Output>,
{
    type Output = _Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE>;
    #[track_caller]
    fn avgpool2d(
        &self,
        kernels_shape: &Shape,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError> {
        let kernel_size: <T as FloatOutBinary>::Output = kernels_shape.size().cast();
        let kernel_size_vec =
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec::splat(kernel_size);
        pooling_template(
            self,
            kernels_shape,
            steps,
            padding,
            dilation,
            |a: T, b: T| a._add(b),
            |a: T::Vec, b: T::Vec| a._add(b),
            |a: T| a._div(kernel_size),
            |a: T::Vec| a._div(kernel_size_vec),
        )
    }

    #[track_caller]
    fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError> {
        adaptive_pooling_template(
            self,
            output_size,
            |a: T, b: T| a._add(b),
            |a: T::Vec, b: T::Vec| a._add(b),
            |a: T, kernel_size: <T as FloatOutBinary>::Output| a._div(kernel_size),
            |a: T::Vec, kernel_size_vec: <<T as FloatOutBinary>::Output as TypeCommon>::Vec| {
                a._div(kernel_size_vec)
            },
        )
    }
}

impl<T, const DEVICE: usize> FloatOutPooling for Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds
        + Cast<T>
        + NormalOut<Output = T>
        + FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
    <T as FloatOutBinary>::Output:
        CommonBounds + FloatOutBinary<Output = <T as FloatOutBinary>::Output>,
    T::Vec: VecTrait<T>
        + Copy
        + Send
        + Sync
        + NormalOut<Output = T::Vec>
        + FloatOutBinary<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
    bool: Cast<T>,
    i64: Cast<<T as FloatOutBinary>::Output>,
{
    type Output = Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE>;
    #[track_caller]
    fn avgpool2d(
        &self,
        kernels_shape: &Shape,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError> {
        Ok(self
            .inner
            .avgpool2d(&kernels_shape, steps, padding, dilation)?
            .into())
    }

    #[track_caller]
    fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError> {
        Ok(self.inner.adaptive_avgpool2d(output_size)?.into())
    }
}
