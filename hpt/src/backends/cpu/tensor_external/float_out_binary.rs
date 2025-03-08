use hpt_traits::{ops::binary::FloatBinOps, tensor::CommonBounds};
use hpt_types::{dtype::TypeCommon, type_promote::FloatOutBinary};

use crate::{backend::Cpu, Tensor};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};

type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE: usize, Al> FloatBinOps for Tensor<T, Cpu, DEVICE, Al>
where
    T: FloatOutBinary + CommonBounds,
    FloatBinaryType<T>: CommonBounds,
    T::Vec: FloatOutBinary<Output = <FloatBinaryType<T> as TypeCommon>::Vec>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<FloatBinaryType<T>, Cpu, DEVICE, Al>;

    type OutputMeta = FloatBinaryType<T>;

    type InplaceOutput = Tensor<FloatBinaryType<T>, Cpu, DEVICE, Al>;

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
