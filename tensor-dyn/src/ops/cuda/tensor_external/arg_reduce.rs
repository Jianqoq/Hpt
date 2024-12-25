use crate::{Cuda, Tensor};
use cudarc::{driver::DeviceRepr, types::CudaTypeName};
use tensor_common::axis::Axis;
use tensor_traits::{CommonBounds, IndexReduce};
use tensor_types::{
    into_scalar::IntoScalar,
    type_promote::{Cmp, NormalOut},
};
impl<
        T: CommonBounds + NormalOut<Output = T> + Cmp + DeviceRepr + CudaTypeName + IntoScalar<i64>,
        const DEVICE_ID: usize,
    > IndexReduce for Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Tensor<i64, Cuda, DEVICE_ID>;

    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.argmax(axis, keep_dims)?.into())
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.argmin(axis, keep_dims)?.into())
    }
}
