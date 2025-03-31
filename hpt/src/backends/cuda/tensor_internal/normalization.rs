use cudarc::driver::DeviceRepr;
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cuda,
};
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{
    ops::normalization::NormalizationOps,
    tensor::{CommonBounds, TensorInfo},
};
use hpt_types::{
    dtype::{CudaType, TypeCommon},
    into_scalar::Cast,
    into_vec::IntoVec,
    type_promote::{FloatOutBinary, FloatOutUnary, FloatOutUnaryPromote, NormalOut},
};

use crate::tensor_base::_Tensor;

use super::softmax::{contiguous_softmax, uncontiguous_softmax};
use super::layernorm::contiguous_layernorm;
type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE: usize, A> NormalizationOps for _Tensor<T, Cuda, DEVICE, A>
where
    T: CommonBounds
        + FloatOutBinary
        + Cast<FloatBinaryType<T>>
        + FloatOutUnary<Output = FloatBinaryType<T>>
        + CudaType
        + DeviceRepr,
    <T as FloatOutUnaryPromote>::Intermediate: DeviceRepr,
    T::Vec: FloatOutUnary<Output = <FloatBinaryType<T> as TypeCommon>::Vec>
        + IntoVec<<FloatBinaryType<T> as TypeCommon>::Vec>,
    FloatBinaryType<T>: CommonBounds
        + FloatOutUnary<Output = FloatBinaryType<T>>
        + NormalOut<T, Output = FloatBinaryType<T>>
        + CudaType
        + DeviceRepr,
    <FloatBinaryType<T> as TypeCommon>::Vec:
        FloatOutUnary<Output = <FloatBinaryType<T> as TypeCommon>::Vec>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<FloatBinaryType<T>, Cuda, DEVICE, A>;

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
        let normalized_shape: Shape = normalized_shape.into();
        contiguous_layernorm(
            self,
            gamma,
            beta,
            eps,
            &normalized_shape,
            None::<Self::Output>,
        )
    }

    fn softmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        let res = if self.is_contiguous() && self.parent().is_none() {
            contiguous_softmax(self, axis, None::<Self::Output>)?
        } else {
            uncontiguous_softmax(self, axis, None::<Self::Output>)?
        };
        Ok(res)
    }

    fn log_softmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }
}
