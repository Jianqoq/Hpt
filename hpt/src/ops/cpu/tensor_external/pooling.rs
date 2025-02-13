use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{CommonBounds, FloatOutPooling, NormalPooling};
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    traits::VecTrait,
    type_promote::{FloatOutBinary, NormalOut},
};

use crate::{Cpu, Tensor};

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
    fn avgpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError> {
        Ok(self
            .inner
            .avgpool2d(kernels_shape, steps, padding, dilation)?
            .into())
    }

    #[track_caller]
    fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError> {
        Ok(self.inner.adaptive_avgpool2d(output_size)?.into())
    }
}

impl<T, const DEVICE: usize> NormalPooling for Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Cast<T> + NormalOut<Output = T>,
    T::Vec: VecTrait<T> + Copy + Send + Sync + NormalOut<Output = T::Vec>,
    bool: Cast<T>,
    i64: Cast<T>,
{
    type Output = Tensor<T, Cpu, DEVICE>;
    #[track_caller]
    fn maxpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError> {
        Ok(self
            .inner
            .maxpool2d(kernels_shape, steps, padding, dilation)?
            .into())
    }

    #[track_caller]
    fn adaptive_maxpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError> {
        Ok(self.inner.adaptive_maxpool2d(output_size)?.into())
    }
}
