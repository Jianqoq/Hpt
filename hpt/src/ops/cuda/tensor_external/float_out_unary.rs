use std::borrow::BorrowMut;

use cudarc::driver::DeviceRepr;
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, FloatUnaryOps};
use hpt_types::dtype::CudaType;
use hpt_types::{
    cuda_types::scalar::Scalar, dtype::TypeCommon, into_scalar::Cast, type_promote::FloatOutUnary,
};

use crate::{
    ops::cpu::tensor_internal::float_out_unary::FloatUnaryType, tensor::Tensor,
    tensor_base::_Tensor, Cuda,
};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};

impl<T, const DEVICE_ID: usize, Al> FloatUnaryOps for Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: FloatOutUnary + CommonBounds + DeviceRepr + CudaType,
    FloatUnaryType<T>: CommonBounds + DeviceRepr + CudaType,
    f64: Cast<<T as FloatOutUnary>::Output>,
    <FloatUnaryType<T> as TypeCommon>::Vec: Send + Copy + Sync,
    Scalar<T>: FloatOutUnary<Output = Scalar<FloatUnaryType<T>>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID, Al>;

    type InplaceOutput = Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID, Al>;

    type OutputMeta = <T as FloatOutUnary>::Output;

    fn sin(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::sin(self.inner.as_ref())?.into())
    }

    fn cos(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::cos(self.inner.as_ref())?.into())
    }

    fn tan(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::tan(self.inner.as_ref())?.into())
    }

    fn asin(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::asin(self.inner.as_ref())?.into())
    }

    fn acos(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::acos(self.inner.as_ref())?.into())
    }

    fn atan(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::atan(self.inner.as_ref())?.into())
    }

    fn sinh(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::sinh(self.inner.as_ref())?.into())
    }

    fn cosh(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::cosh(self.inner.as_ref())?.into())
    }

    fn tanh(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::tanh(self.inner.as_ref())?.into())
    }

    fn asinh(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::asinh(self.inner.as_ref())?.into())
    }

    fn acosh(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::acosh(self.inner.as_ref())?.into())
    }

    fn atanh(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::atanh(self.inner.as_ref())?.into())
    }

    fn sin_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::sin_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn cos_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::cos_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn tan_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::tan_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn asin_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::asin_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn acos_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::acos_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn atan_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::atan_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sinh_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::sinh_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn cosh_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::cosh_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn tanh_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::tanh_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn asinh_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::asinh_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn acosh_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::acosh_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn atanh_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::atanh_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn exp(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::exp(self.inner.as_ref())?.into())
    }

    fn exp_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::exp_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn exp2(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::exp2(self.inner.as_ref())?.into())
    }

    fn exp2_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::exp2_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sqrt(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::sqrt(self.inner.as_ref())?.into())
    }

    fn sqrt_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::sqrt_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn recip(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::recip(self.inner.as_ref())?.into())
    }

    fn recip_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::recip_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn ln(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::ln(self.inner.as_ref())?.into())
    }

    fn ln_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::ln_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn log2(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::log2(self.inner.as_ref())?.into())
    }

    fn log2_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::log2_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn log10(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::log10(self.inner.as_ref())?.into())
    }

    fn log10_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::log10_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn celu(&self, alpha: Self::OutputMeta) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::celu(self.inner.as_ref(), alpha)?.into())
    }

    fn celu_<U>(&self, alpha: Self::OutputMeta, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::celu_(
            self.inner.as_ref(),
            alpha,
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sigmoid(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::sigmoid(self.inner.as_ref())?.into())
    }

    fn sigmoid_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::sigmoid_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn elu(&self, alpha: Self::OutputMeta) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::elu(self.inner.as_ref(), alpha)?.into())
    }

    fn elu_<U>(&self, alpha: Self::OutputMeta, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::elu_(
            self.inner.as_ref(),
            alpha,
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn erf(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::erf(self.inner.as_ref())?.into())
    }

    fn gelu(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::gelu(self.inner.as_ref())?.into())
    }

    fn gelu_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::gelu_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn selu<U>(&self, alpha: U, gamma: U) -> Result<Self::Output, TensorError>
    where
        U: Into<Option<Self::OutputMeta>>,
    {
        Ok(_Tensor::selu(self.inner.as_ref(), alpha, gamma)?.into())
    }

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        mut out: U,
    ) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::selu_(
            self.inner.as_ref(),
            alpha,
            gamma,
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn hard_sigmoid(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::hard_sigmoid(self.inner.as_ref())?.into())
    }

    fn hard_sigmoid_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::hard_sigmoid_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn hard_swish(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::hard_swish(self.inner.as_ref())?.into())
    }

    fn hard_swish_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::hard_swish_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn softplus(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::softplus(self.inner.as_ref())?.into())
    }

    fn softplus_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::softplus_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn softsign(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::softsign(self.inner.as_ref())?.into())
    }

    fn softsign_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::softsign_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn mish(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::mish(self.inner.as_ref())?.into())
    }

    fn mish_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::mish_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn cbrt(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::cbrt(self.inner.as_ref())?.into())
    }

    fn cbrt_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::cbrt_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sincos(&self) -> Result<(Self::Output, Self::Output), TensorError> {
        let (sin, cos) = _Tensor::sincos(self.inner.as_ref())?;
        Ok((sin.into(), cos.into()))
    }

    fn sincos_<U, O>(&self, mut outs: (U, O)) -> Result<(Self::Output, Self::Output), TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
        O: BorrowMut<Self::InplaceOutput>,
    {
        let (sin, cos) = _Tensor::sincos_(
            self.inner.as_ref(),
            (
                &mut outs.0.borrow_mut().inner.as_ref().clone(),
                &mut outs.1.borrow_mut().inner.as_ref().clone(),
            ),
        )?;
        Ok((sin.into(), cos.into()))
    }

    fn exp10(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::exp10(self.inner.as_ref())?.into())
    }

    fn exp10_<U>(&self, mut out: U) -> Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::exp10_(
            self.inner.as_ref(),
            &mut out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn erf_<U>(&self, mut out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::erf_(self.inner.as_ref(), out.borrow_mut().inner.as_ref().clone())?.into())
    }
}
