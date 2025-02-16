use std::borrow::BorrowMut;

use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, FloatUnaryOps};
use hpt_types::{dtype::TypeCommon, into_scalar::Cast, type_promote::FloatOutUnary};

use crate::{
    backend::Cpu, ops::cpu::tensor_internal::float_out_unary::FloatUnaryType, tensor::Tensor,
    tensor_base::_Tensor,
};

impl<T, const DEVICE: usize> FloatUnaryOps for Tensor<T, Cpu, DEVICE>
where
    T: FloatOutUnary + CommonBounds,
    FloatUnaryType<T>: CommonBounds,
    f64: Cast<<T as FloatOutUnary>::Output>,
    T::Vec: FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec>,
{
    type Output = Tensor<FloatUnaryType<T>, Cpu, DEVICE>;

    type InplaceOutput = Tensor<FloatUnaryType<T>, Cpu, DEVICE>;

    type OutputMeta = FloatUnaryType<T>;

    fn sin(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::sin(self.inner.as_ref())?.into())
    }

    fn cos(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::cos(self.inner.as_ref())?.into())
    }

    fn tan(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::tan(self.inner.as_ref())?.into())
    }

    fn asin(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::asin(self.inner.as_ref())?.into())
    }

    fn acos(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::acos(self.inner.as_ref())?.into())
    }

    fn atan(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::atan(self.inner.as_ref())?.into())
    }

    fn sinh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::sinh(self.inner.as_ref())?.into())
    }

    fn cosh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::cosh(self.inner.as_ref())?.into())
    }

    fn tanh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::tanh(self.inner.as_ref())?.into())
    }

    fn asinh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::asinh(self.inner.as_ref())?.into())
    }

    fn acosh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::acosh(self.inner.as_ref())?.into())
    }

    fn atanh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::atanh(self.inner.as_ref())?.into())
    }

    fn sin_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::sin_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn cos_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::cos_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn tan_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::tan_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn asin_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::asin_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn acos_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::acos_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn atan_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::atan_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sinh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::sinh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn cosh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::cosh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn tanh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::tanh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn asinh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::asinh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn acosh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::acosh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn atanh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::atanh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn exp(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::exp(self.inner.as_ref())?.into())
    }

    fn exp_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::exp_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn exp2(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::exp2(self.inner.as_ref())?.into())
    }

    fn exp2_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::exp2_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sqrt(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::sqrt(self.inner.as_ref())?.into())
    }

    fn sqrt_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::sqrt_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn recip(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::recip(self.inner.as_ref())?.into())
    }

    fn recip_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::recip_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn ln(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::ln(self.inner.as_ref())?.into())
    }

    fn ln_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::ln_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn log2(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::log2(self.inner.as_ref())?.into())
    }

    fn log2_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::log2_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn log10(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::log10(self.inner.as_ref())?.into())
    }

    fn log10_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::log10_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn celu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::celu(self.inner.as_ref(), alpha)?.into())
    }

    fn celu_<U>(
        &self,
        alpha: Self::OutputMeta,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::celu_(
            self.inner.as_ref(),
            alpha,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::sigmoid(self.inner.as_ref())?.into())
    }

    fn sigmoid_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::sigmoid_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn elu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::elu(self.inner.as_ref(), alpha)?.into())
    }

    fn elu_<U>(
        &self,
        alpha: Self::OutputMeta,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::elu_(
            self.inner.as_ref(),
            alpha,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn erf(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::erf(self.inner.as_ref())?.into())
    }

    fn gelu(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::gelu(self.inner.as_ref())?.into())
    }

    fn gelu_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::gelu_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn selu<U>(&self, alpha: U, gamma: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Into<Option<Self::OutputMeta>>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::selu(self.inner.as_ref(), alpha, gamma)?.into())
    }

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::selu_(
            self.inner.as_ref(),
            alpha,
            gamma,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::hard_sigmoid(self.inner.as_ref())?.into())
    }

    fn hard_sigmoid_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::hard_sigmoid_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::hard_swish(self.inner.as_ref())?.into())
    }

    fn hard_swish_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::hard_swish_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn softplus(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::softplus(self.inner.as_ref())?.into())
    }

    fn softplus_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::softplus_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn softsign(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::softsign(self.inner.as_ref())?.into())
    }

    fn softsign_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::softsign_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn mish(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::mish(self.inner.as_ref())?.into())
    }

    fn mish_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::mish_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::cbrt(self.inner.as_ref())?.into())
    }

    fn cbrt_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::cbrt_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sincos(&self) -> std::result::Result<(Self::Output, Self::Output), TensorError> {
        let (sin, cos) = self.inner.sincos()?;
        Ok((sin.into(), cos.into()))
    }

    fn exp10(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::exp10(self.inner.as_ref())?.into())
    }

    fn exp10_<U>(&self, mut out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::exp10_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sincos_<U, O>(
        &self,
        mut outs: (U, O),
    ) -> std::result::Result<(Self::Output, Self::Output), TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
        O: BorrowMut<Self::InplaceOutput>,
    {
        let (sin, cos) = self.inner.sincos_((
            outs.0.borrow_mut().inner.as_ref().clone(),
            outs.1.borrow_mut().inner.as_ref().clone(),
        ))?;
        Ok((sin.into(), cos.into()))
    }

    fn erf_<U>(&self, mut out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::erf_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }
}
