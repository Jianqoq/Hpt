use crate::Tensor;
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_common::error::base::TensorError;
use hpt_traits::{ops::cumulative::CumulativeOps, tensor::CommonBounds};

impl<T: CommonBounds, const DEVICE: usize, Al> CumulativeOps for Tensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    fn cumsum<A: Into<Option<i64>>>(&self, axis: A) -> Result<Self, TensorError> {
        Ok(self.inner.cumsum(axis)?.into())
    }
    fn cumprod<A: Into<Option<i64>>>(&self, axis: A) -> Result<Self, TensorError> {
        Ok(self.inner.cumprod(axis)?.into())
    }
}
