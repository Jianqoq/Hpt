use hpt_traits::{CommonBounds, FloatBinOps};
use hpt_types::{dtype::TypeCommon, type_promote::FloatOutBinary};

use crate::{
    ops::cpu::tensor_internal::float_out_unary::FloatBinaryType, Cpu, Tensor,
};

impl<T, const DEVICE: usize> FloatBinOps for Tensor<T, Cpu, DEVICE>
where
    T: FloatOutBinary + CommonBounds,
    FloatBinaryType<T>: CommonBounds,
    T::Vec: FloatOutBinary<Output = <FloatBinaryType<T> as TypeCommon>::Vec>,
{
    type Output = Tensor<FloatBinaryType<T>, Cpu, DEVICE>;

    type OutputMeta = FloatBinaryType<T>;

    type InplaceOutput = Tensor<FloatBinaryType<T>, Cpu, DEVICE>;

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
