use crate::Tensor;
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cuda;
use hpt_common::axis::axis::Axis;
use hpt_common::error::base::TensorError;
use hpt_traits::{ops::reduce::IndexReduce, tensor::CommonBounds};
use hpt_types::{
    dtype::CudaType,
    into_scalar::Cast,
    type_promote::{Cmp, NormalOut},
};

impl<
        T: CommonBounds + NormalOut<Output = T> + Cmp + DeviceRepr + CudaType + Cast<i64>,
        const DEVICE_ID: usize,
        Al,
    > IndexReduce for Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<i64, Cuda, DEVICE_ID, Al>;

    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.argmax(axis, keep_dims)?.into())
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.argmin(axis, keep_dims)?.into())
    }
}
