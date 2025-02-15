use crate::Cpu;
use crate::Tensor;
use hpt_common::error::base::TensorError;
use hpt_traits::CommonBounds;
use hpt_traits::CumulativeOps;

impl<T: CommonBounds, const DEVICE: usize> CumulativeOps for Tensor<T, Cpu, DEVICE> {
    fn cumsum<A: Into<Option<i64>>>(&self, axis: A) -> Result<Self, TensorError> {
        Ok(self.inner.cumsum(axis)?.into())
    }
    fn cumprod<A: Into<Option<i64>>>(&self, axis: A) -> Result<Self, TensorError> {
        Ok(self.inner.cumprod(axis)?.into())
    }
}
