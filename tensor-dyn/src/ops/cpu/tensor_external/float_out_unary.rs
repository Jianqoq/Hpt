use std::borrow::Borrow;

use tensor_common::err_handler::ErrHandler;
use tensor_traits::{CommonBounds, FloatUaryOps};
use tensor_types::{dtype::TypeCommon, into_scalar::IntoScalar, type_promote::FloatOutUnary};

use crate::{
    backend::Cpu, ops::cpu::tensor_internal::float_out_unary::FloatUnaryType, tensor::Tensor,
    tensor_base::_Tensor,
};

impl<T> FloatUaryOps for Tensor<T>
where
    T: FloatOutUnary + CommonBounds,
    FloatUnaryType<T>: CommonBounds,
    f64: IntoScalar<<T as FloatOutUnary>::Output>,
    T::Vec: FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec>,
{
    type Output = Tensor<FloatUnaryType<T>>;

    type InplaceOutput = Tensor<FloatUnaryType<T>>;

    type OutputMeta = FloatUnaryType<T>;

    fn sin(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::sin(self.inner.as_ref())?.into())
    }

    fn cos(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::cos(self.inner.as_ref())?.into())
    }

    fn tan(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::tan(self.inner.as_ref())?.into())
    }

    fn asin(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::asin(self.inner.as_ref())?.into())
    }

    fn acos(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::acos(self.inner.as_ref())?.into())
    }

    fn atan(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::atan(self.inner.as_ref())?.into())
    }

    fn sinh(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::sinh(self.inner.as_ref())?.into())
    }

    fn cosh(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::cosh(self.inner.as_ref())?.into())
    }

    fn tanh(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::tanh(self.inner.as_ref())?.into())
    }

    fn asinh(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::asinh(self.inner.as_ref())?.into())
    }

    fn acosh(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::acosh(self.inner.as_ref())?.into())
    }

    fn atanh(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::atanh(self.inner.as_ref())?.into())
    }

    fn sin_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::sin_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn cos_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::cos_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn tan_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::tan_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn asin_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::asin_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn acos_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::acos_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn atan_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::atan_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn sinh_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::sinh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn cosh_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::cosh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn tanh_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::tanh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn asinh_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::asinh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn acosh_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::acosh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn atanh_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::atanh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn exp(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::exp(self.inner.as_ref())?.into())
    }

    fn exp_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::exp_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn exp2(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::exp2(self.inner.as_ref())?.into())
    }

    fn exp2_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::exp2_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn sqrt(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::sqrt(self.inner.as_ref())?.into())
    }

    fn sqrt_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::sqrt_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn recip(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::recip(self.inner.as_ref())?.into())
    }

    fn recip_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::recip_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn ln(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::ln(self.inner.as_ref())?.into())
    }

    fn ln_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::ln_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn log2(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::log2(self.inner.as_ref())?.into())
    }

    fn log2_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::log2_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn log10(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::log10(self.inner.as_ref())?.into())
    }

    fn log10_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::log10_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn celu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::celu(self.inner.as_ref(), alpha)?.into())
    }

    fn celu_<U>(
        &self,
        alpha: Self::OutputMeta,
        out: U,
    ) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(
            _Tensor::<T, Cpu>::celu_(self.inner.as_ref(), alpha, out.borrow().inner.as_ref())?
                .into(),
        )
    }

    fn sigmoid(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::sigmoid(self.inner.as_ref())?.into())
    }

    fn sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::sigmoid_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn elu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::elu(self.inner.as_ref(), alpha)?.into())
    }

    fn elu_<U>(
        &self,
        alpha: Self::OutputMeta,
        out: U,
    ) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(
            _Tensor::<T, Cpu>::elu_(self.inner.as_ref(), alpha, out.borrow().inner.as_ref())?
                .into(),
        )
    }

    fn erf(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::erf(self.inner.as_ref())?.into())
    }

    fn fast_hard_sigmoid(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::fast_hard_sigmoid(self.inner.as_ref())?.into())
    }

    fn gelu(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::gelu(self.inner.as_ref())?.into())
    }

    fn gelu_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::gelu_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn selu<U>(&self, alpha: U, gamma: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Into<Option<Self::OutputMeta>>,
    {
        Ok(_Tensor::<T, Cpu>::selu(self.inner.as_ref(), alpha, gamma)?.into())
    }

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U,
    ) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::selu_(
            self.inner.as_ref(),
            alpha,
            gamma,
            out.borrow().inner.as_ref(),
        )?
        .into())
    }

    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::hard_sigmoid(self.inner.as_ref())?.into())
    }

    fn hard_sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(
            _Tensor::<T, Cpu>::hard_sigmoid_(self.inner.as_ref(), out.borrow().inner.as_ref())?
                .into(),
        )
    }

    fn hard_swish(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::hard_swish(self.inner.as_ref())?.into())
    }

    fn hard_swish_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(
            _Tensor::<T, Cpu>::hard_swish_(self.inner.as_ref(), out.borrow().inner.as_ref())?
                .into(),
        )
    }

    fn softplus(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::softplus(self.inner.as_ref())?.into())
    }

    fn softplus_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::softplus_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn softsign(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::softsign(self.inner.as_ref())?.into())
    }

    fn softsign_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::softsign_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn mish(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::mish(self.inner.as_ref())?.into())
    }

    fn mish_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::mish_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn cbrt(&self) -> std::result::Result<Self::Output, ErrHandler> {
        Ok(_Tensor::<T, Cpu>::cbrt(self.inner.as_ref())?.into())
    }

    fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::Output, ErrHandler>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::cbrt_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }
}
