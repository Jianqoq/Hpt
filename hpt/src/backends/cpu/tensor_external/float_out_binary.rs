use hpt_traits::{ops::binary::FloatBinOps, tensor::CommonBounds};
use hpt_types::{dtype::TypeCommon, type_promote::FloatOutBinary};

use crate::{backend::Cpu, Tensor};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};

type FloatBinaryType<T, B> = <T as FloatOutBinary<B>>::Output;

impl<T, B, const DEVICE: usize, Al> FloatBinOps<Tensor<B, Cpu, DEVICE, Al>>
    for Tensor<T, Cpu, DEVICE, Al>
where
    B: CommonBounds,
    T: FloatOutBinary<B> + CommonBounds,
    FloatBinaryType<T, B>: CommonBounds,
    T::Vec: FloatOutBinary<B::Vec, Output = <FloatBinaryType<T, B> as TypeCommon>::Vec>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<FloatBinaryType<T, B>, Cpu, DEVICE, Al>;

    type OutputMeta = FloatBinaryType<T, B>;

    type InplaceOutput = Tensor<FloatBinaryType<T, B>, Cpu, DEVICE, Al>;

    fn hypot<C>(
        &self,
        rhs: C,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<Tensor<B, Cpu, DEVICE, Al>>,
    {
        Ok(self.inner.as_ref().hypot(rhs.into().inner.as_ref())?.into())
    }

    fn hypot_<C, U>(
        &self,
        rhs: C,
        mut out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<Tensor<B, Cpu, DEVICE, Al>>,
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        Ok(self
            .inner
            .as_ref()
            .hypot_(
                rhs.into().inner.as_ref(),
                out.borrow_mut().inner.as_ref().clone(),
            )?
            .into())
    }

    fn div_<C, U>(
        &self,
        rhs: C,
        mut out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<Tensor<B, Cpu, DEVICE, Al>>,
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        Ok(self
            .inner
            .as_ref()
            .div_(
                rhs.into().inner.as_ref(),
                out.borrow_mut().inner.as_ref().clone(),
            )?
            .into())
    }

    fn pow<C>(
        &self,
        rhs: C,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<Tensor<B, Cpu, DEVICE, Al>>,
    {
        Ok(self.inner.as_ref().pow(rhs.into().inner.as_ref())?.into())
    }

    fn pow_<C, U>(
        &self,
        rhs: C,
        mut out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<Tensor<B, Cpu, DEVICE, Al>>,
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        Ok(self
            .inner
            .as_ref()
            .pow_(
                rhs.into().inner.as_ref(),
                out.borrow_mut().inner.as_ref().clone(),
            )?
            .into())
    }
}
