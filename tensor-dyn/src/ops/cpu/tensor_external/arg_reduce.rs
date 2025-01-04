use tensor_common::axis::axis::Axis;
use tensor_common::error::base::TensorError;
use tensor_traits::{ CommonBounds, IndexReduce };
use tensor_types::type_promote::{ Cmp, NormalOut };

use crate::tensor::Tensor;

impl<T: CommonBounds + NormalOut<Output = T> + Cmp<T, Output = bool>> IndexReduce for Tensor<T> {
    type Output = Tensor<i64>;

    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.argmax(axis, keep_dims)?.into())
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.argmin(axis, keep_dims)?.into())
    }
}
