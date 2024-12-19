use tensor_common::axis::Axis;
use tensor_traits::{ CommonBounds, IndexReduce };
use tensor_types::type_promote::{ Cmp, NormalOut };

use crate::tensor::Tensor;

impl<T: CommonBounds + NormalOut<Output = T> + Cmp<T, Output = bool>> IndexReduce for Tensor<T> {
    type Output = Tensor<i64>;

    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.argmax(axis, keep_dims)?.into())
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.argmin(axis, keep_dims)?.into())
    }
}
