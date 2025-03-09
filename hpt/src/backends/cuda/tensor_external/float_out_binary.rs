use crate::{backend::Cuda, Tensor};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_traits::{ops::binary::FloatBinOps, tensor::CommonBounds};
use hpt_types::{cuda_types::scalar::Scalar, dtype::CudaType, type_promote::FloatOutBinary};

type FloatBinaryType<T, B> = <T as FloatOutBinary<B>>::Output;

impl<T, B, const DEVICE: usize, Al> FloatBinOps<Tensor<B, Cuda, DEVICE, Al>>
    for Tensor<T, Cuda, DEVICE, Al>
where
    B: CommonBounds + DeviceRepr + CudaType,
    T: FloatOutBinary<B> + CommonBounds + DeviceRepr + CudaType,
    FloatBinaryType<T, B>: CommonBounds + DeviceRepr + CudaType,
    Scalar<T>: FloatOutBinary<Scalar<B>, Output = Scalar<FloatBinaryType<T, B>>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<FloatBinaryType<T, B>, Cuda, DEVICE, Al>;

    type OutputMeta = FloatBinaryType<T, B>;

    type InplaceOutput = Tensor<FloatBinaryType<T, B>, Cuda, DEVICE, Al>;

    fn hypot<C>(
        &self,
        rhs: C,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<Tensor<B, Cuda, DEVICE, Al>>,
    {
        Ok(self.inner.as_ref().hypot(rhs.into().inner.as_ref())?.into())
    }

    fn hypot_<C, U>(
        &self,
        rhs: C,
        mut out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        C: Into<Tensor<B, Cuda, DEVICE, Al>>,
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
        C: Into<Tensor<B, Cuda, DEVICE, Al>>,
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
        C: Into<Tensor<B, Cuda, DEVICE, Al>>,
    {
        Ok(self.inner.as_ref().pow(rhs.into().inner.as_ref())?.into())
    }

    fn pow_<C, U>(
        &self,
        rhs: C,
        mut out: U,
    ) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
        C: Into<Tensor<B, Cuda, DEVICE, Al>>,
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
