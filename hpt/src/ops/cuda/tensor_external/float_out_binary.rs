use cudarc::driver::DeviceRepr;
use hpt_traits::{CommonBounds, FloatBinOps};
use hpt_types::{cuda_types::scalar::Scalar, dtype::CudaType, type_promote::FloatOutBinary};

use crate::{ops::cpu::tensor_internal::float_out_unary::FloatBinaryType, Cuda, Tensor};

impl<T, const DEVICE: usize> FloatBinOps for Tensor<T, Cuda, DEVICE>
where
    T: CommonBounds + DeviceRepr + CudaType + FloatOutBinary,
    Scalar<T>: FloatOutBinary<Output = Scalar<FloatBinaryType<T>>>,
    FloatBinaryType<T>: CommonBounds + DeviceRepr + CudaType,
{
    type Output = Tensor<FloatBinaryType<T>, Cuda, DEVICE>;

    type OutputMeta = FloatBinaryType<T>;

    type InplaceOutput = Tensor<FloatBinaryType<T>, Cuda, DEVICE>;

    fn hypot(
        &self,
        rhs: &Self,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError> {
        Ok(self.inner.as_ref().hypot(rhs.inner.as_ref())?.into())
    }

    fn hypot_<U>(
        &self,
        rhs: &Self,
        mut out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        Ok(self
            .inner
            .as_ref()
            .hypot_(rhs.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?
            .into())
    }

    fn div_<U>(
        &self,
        rhs: &Self,
        mut out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        Ok(self
            .inner
            .as_ref()
            .div_(rhs.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?
            .into())
    }
}
