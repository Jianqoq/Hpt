use crate::{Cuda, Tensor};
use cudarc::driver::DeviceRepr;
use hpt_common::axis::axis::Axis;
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, IndexReduce};
use hpt_types::{
    dtype::CudaType,
    into_scalar::Cast,
    type_promote::{Cmp, NormalOut},
};
impl<
        T: CommonBounds + NormalOut<Output = T> + Cmp + DeviceRepr + CudaType + Cast<i64>,
        const DEVICE_ID: usize,
    > IndexReduce for Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Tensor<i64, Cuda, DEVICE_ID>;

    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.argmax(axis, keep_dims)?.into())
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output, TensorError> {
        Ok(self.inner.argmin(axis, keep_dims)?.into())
    }
}
