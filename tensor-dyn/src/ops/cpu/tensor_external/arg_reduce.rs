use tensor_common::axis::axis::Axis;
use tensor_common::error::base::TensorError;
use tensor_traits::{CommonBounds, IndexReduce};
use tensor_types::type_promote::{Cmp, NormalOut};

use crate::{
    tensor::{DiffTensor, Tensor},
    Cpu,
};

impl<T: CommonBounds + NormalOut<Output = T> + Cmp<T, Output = bool>, const DEVICE: usize>
    IndexReduce for Tensor<T, Cpu, DEVICE>
{
    type Output = Tensor<i64, Cpu, DEVICE>;

    fn argmax<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.argmax(axis, keep_dims)?.into())
    }

    fn argmin<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.argmin(axis, keep_dims)?.into())
    }
}

impl<T: CommonBounds + NormalOut<Output = T> + Cmp<T, Output = bool>, const DEVICE: usize>
    IndexReduce for DiffTensor<T, Cpu, DEVICE>
{
    type Output = Tensor<i64, Cpu, DEVICE>;

    fn argmax<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.argmax(axis, keep_dims)?.into())
    }

    fn argmin<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.argmin(axis, keep_dims)?.into())
    }
}
