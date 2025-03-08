use hpt_allocator::Cpu;
use hpt_common::axis::axis::Axis;
use hpt_common::error::base::TensorError;
use hpt_traits::{ops::reduce::IndexReduce, tensor::CommonBounds};
use hpt_types::type_promote::{Cmp, NormalOut};

use crate::tensor::{DiffTensor, Tensor};

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
