use crate::{Cuda, Tensor};
use cudarc::{driver::DeviceRepr, types::CudaTypeName};
use tensor_common::axis::Axis;
use tensor_common::err_handler::TensorError;
use tensor_traits::{CommonBounds, IndexReduce};
use tensor_types::{
    cast::Cast,
    type_promote::{Cmp, NormalOut},
};
impl<
        T: CommonBounds + NormalOut<Output = T> + Cmp + DeviceRepr + CudaTypeName + Cast<i64>,
        const DEVICE_ID: usize,
    > IndexReduce for Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Tensor<i64, Cuda, DEVICE_ID>;

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
