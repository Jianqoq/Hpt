use crate::backend::Cpu;
use crate::Tensor;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_common::shape::shape::Shape;
use hpt_traits::ops::normalization::NormalizationOps;
use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::into_vec::IntoVec;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::FloatOutUnary;
use hpt_types::type_promote::NormalOut;

type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE: usize, A> NormalizationOps for Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds
        + FloatOutBinary
        + Cast<FloatBinaryType<T>>
        + FloatOutUnary<Output = FloatBinaryType<T>>,
    T::Vec: FloatOutUnary<Output = <FloatBinaryType<T> as TypeCommon>::Vec>
        + IntoVec<<FloatBinaryType<T> as TypeCommon>::Vec>,
    FloatBinaryType<T>: CommonBounds
        + FloatOutUnary<Output = FloatBinaryType<T>>
        + NormalOut<T, Output = FloatBinaryType<T>>,
    <FloatBinaryType<T> as TypeCommon>::Vec:
        FloatOutUnary<Output = <FloatBinaryType<T> as TypeCommon>::Vec>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<FloatBinaryType<T>, Cpu, DEVICE, A>;

    type OutputMeta = FloatBinaryType<T>;

    fn layernorm<S: Into<Shape>>(
        &self,
        normalized_shape: S,
        gamma: Option<&Self::Output>,
        beta: Option<&Self::Output>,
        eps: Self::OutputMeta,
    ) -> Result<Self::Output, TensorError>
    where
        usize: Cast<Self::OutputMeta>,
    {
        Ok(self
            .inner
            .layernorm(
                normalized_shape,
                gamma.map(|t| t.inner.as_ref()),
                beta.map(|t| t.inner.as_ref()),
                eps,
            )?
            .into())
    }

    fn softmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        Ok(self.inner.softmax(axis)?.into())
    }

    fn log_softmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        Ok(self.inner.log_softmax(axis)?.into())
    }
}
