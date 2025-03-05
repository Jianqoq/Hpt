use std::borrow::BorrowMut;

use crate::{
    ops::cpu::tensor_internal::normal_out_unary::NormalType, tensor::Tensor, tensor_base::_Tensor,
    Cpu,
};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, NormalUaryOps, TensorLike};
use hpt_types::type_promote::NormalOutUnary;

impl<T, const DEVICE: usize, Al> NormalUaryOps for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds,
    NormalType<T>: CommonBounds,
    T::Vec: NormalOutUnary,
    _Tensor<NormalType<T>, Cpu, DEVICE, Al>: TensorLike<NormalType<T>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<NormalType<T>, Cpu, DEVICE, Al>;

    type InplaceOutput = Tensor<NormalType<T>, Cpu, DEVICE, Al>;

    type OutputMeta = NormalType<T>;

    fn floor(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::floor(self.inner.as_ref())?.into())
    }

    fn floor_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::floor_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn square(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::square(self.inner.as_ref())?.into())
    }

    fn square_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::square_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn abs(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::abs(self.inner.as_ref())?.into())
    }

    fn abs_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::abs_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn ceil(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::ceil(self.inner.as_ref())?.into())
    }

    fn ceil_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::ceil_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sign(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sign(self.inner.as_ref())?.into())
    }

    fn sign_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sign_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn clamp(
        &self,
        min: Self::OutputMeta,
        max: Self::OutputMeta,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::clamp(self.inner.as_ref(), min, max)?.into())
    }

    fn clamp_<U>(
        &self,
        min: Self::OutputMeta,
        max: Self::OutputMeta,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::clamp_(
            self.inner.as_ref(),
            min,
            max,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn round(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::round(self.inner.as_ref())?.into())
    }

    fn round_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::round_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn neg(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::neg(self.inner.as_ref())?.into())
    }

    fn neg_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::neg_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn relu(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::relu(self.inner.as_ref())?.into())
    }

    fn relu_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::relu_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn leaky_relu(
        &self,
        alpha: Self::OutputMeta,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::leaky_relu(self.inner.as_ref(), alpha)?.into())
    }

    fn leaky_relu_<U>(
        &self,
        alpha: Self::OutputMeta,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::leaky_relu_(
            self.inner.as_ref(),
            alpha,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn relu6(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::relu6(self.inner.as_ref())?.into())
    }

    fn relu6_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::relu6_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }
}
