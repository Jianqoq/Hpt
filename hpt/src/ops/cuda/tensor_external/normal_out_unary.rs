use std::borrow::BorrowMut;

use crate::{
    ops::cpu::tensor_internal::normal_out_unary::NormalType, tensor::Tensor, tensor_base::_Tensor,
    Cuda,
};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, NormalUaryOps, TensorLike};
use hpt_types::dtype::CudaType;
use hpt_types::{
    cuda_types::scalar::Scalar,
    into_scalar::Cast,
    type_promote::{NormalOut, NormalOutUnary},
};
impl<T, const DEVICE: usize, Al> NormalUaryOps for Tensor<T, Cuda, DEVICE, Al>
where
    T: CommonBounds + Cast<T> + DeviceRepr + CudaType,
    NormalType<T>: CommonBounds + CudaType,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    T::Vec: NormalOutUnary,
    T: NormalOutUnary,
    _Tensor<NormalType<T>, Cuda, DEVICE, Al>: TensorLike<NormalType<T>>,
    Scalar<T>: NormalOutUnary + NormalOut<Output = Scalar<NormalType<T>>>,
{
    type Output = Tensor<NormalType<T>, Cuda, DEVICE, Al>;

    type InplaceOutput = Tensor<NormalType<T>, Cuda, DEVICE, Al>;

    type OutputMeta = NormalType<T>;

    fn floor(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::floor(self.inner.as_ref())?.into())
    }

    fn floor_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::floor_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }

    fn square(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::square(self.inner.as_ref())?.into())
    }

    fn square_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::square_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }

    fn abs(&self) -> std::result::Result<Self, TensorError> {
        Ok(_Tensor::abs(self.inner.as_ref())?.into())
    }

    fn abs_<U>(&self, mut out: U) -> std::result::Result<Self, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::abs_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }

    fn ceil(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::ceil(self.inner.as_ref())?.into())
    }

    fn ceil_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::ceil_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }

    fn sign(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::sign(self.inner.as_ref())?.into())
    }

    fn sign_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::sign_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }

    fn clamp(
        &self,
        min: Self::OutputMeta,
        max: Self::OutputMeta,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::clamp(self.inner.as_ref(), min, max)?.into())
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
        Ok(_Tensor::clamp_(
            self.inner.as_ref(),
            min,
            max,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn round(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::round(self.inner.as_ref())?.into())
    }

    fn round_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::round_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }

    fn neg(&self) -> std::result::Result<Self, TensorError> {
        Ok(_Tensor::neg(self.inner.as_ref())?.into())
    }

    fn neg_<U>(&self, mut out: U) -> std::result::Result<Self, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::neg_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }

    fn relu(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::relu(self.inner.as_ref())?.into())
    }

    fn relu_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::relu_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }

    fn leaky_relu(
        &self,
        alpha: Self::OutputMeta,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::leaky_relu(self.inner.as_ref(), alpha)?.into())
    }

    fn leaky_relu_<U>(
        &self,
        alpha: Self::OutputMeta,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::leaky_relu_(
            self.inner.as_ref(),
            alpha,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn relu6(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::relu6(self.inner.as_ref())?.into())
    }

    fn relu6_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::relu6_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }
}
