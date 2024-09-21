use std::borrow::Borrow;

use tensor_traits::{CommonBounds, FloatUaryOps};
use tensor_types::{dtype::TypeCommon, into_scalar::IntoScalar, type_promote::FloatOutUnary};

use crate::{backend::Cpu, ops::cpu::unary::FloatUnaryType, tensor::Tensor, tensor_base::_Tensor};
use anyhow::Result;

impl<T> FloatUaryOps for Tensor<T>
where
    T: FloatOutUnary<Base = FloatUnaryType<T>> + CommonBounds,
    FloatUnaryType<T>: CommonBounds,
    f64: IntoScalar<<T as FloatOutUnary>::Output>,
    T::Vec:
        FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec, Base = FloatUnaryType<T>>,
    <FloatUnaryType<T> as TypeCommon>::Vec: Send + Copy + Sync,
{
    type Output = Tensor<FloatUnaryType<T>>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>>;

    type OutputMeta = <T as FloatOutUnary>::Base;

    fn erf(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::erf(self)?.into())
    }

    fn fast_hard_sigmoid(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::fast_hard_sigmoid(self)?.into())
    }

    fn relu(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::relu(self)?.into())
    }

    fn relu_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: std::borrow::Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::relu_(self, out.borrow())?.into())
    }

    fn sin(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::sin(self)?.into())
    }

    fn cos(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::cos(self)?.into())
    }

    fn tan(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::tan(self)?.into())
    }

    fn asin(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::asin(self)?.into())
    }

    fn acos(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::acos(self)?.into())
    }

    fn atan(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::atan(self)?.into())
    }

    fn sinh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::sinh(self)?.into())
    }

    fn cosh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::cosh(self)?.into())
    }

    fn tanh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::tanh(self)?.into())
    }

    fn asinh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::asinh(self)?.into())
    }

    fn acosh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::acosh(self)?.into())
    }

    fn atanh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::atanh(self)?.into())
    }

    fn sin_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::sin_(self, out)?.into())
    }

    fn cos_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::cos_(self, out)?.into())
    }

    fn tan_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::tan_(self, out)?.into())
    }

    fn asin_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::asin_(self, out)?.into())
    }

    fn acos_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::acos_(self, out)?.into())
    }

    fn atan_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::atan_(self, out)?.into())
    }

    fn sinh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::sinh_(self, out)?.into())
    }

    fn cosh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::cosh_(self, out)?.into())
    }

    fn tanh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::tanh_(self, out)?.into())
    }

    fn asinh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::asinh_(self, out)?.into())
    }

    fn acosh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::acosh_(self, out)?.into())
    }

    fn atanh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::atanh_(self, out)?.into())
    }

    fn exp(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::exp(self)?.into())
    }

    fn exp_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::exp_(self, out)?.into())
    }

    fn exp2(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::exp2(self)?.into())
    }

    fn exp2_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::exp2_(self, out)?.into())
    }

    fn sqrt(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::sqrt(self)?.into())
    }

    fn sqrt_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::sqrt_(self, out)?.into())
    }

    fn recip(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::recip(self)?.into())
    }

    fn recip_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::recip_(self, out)?.into())
    }

    fn ln(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::ln(self)?.into())
    }

    fn ln_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::ln_(self, out)?.into())
    }

    fn log2(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::log2(self)?.into())
    }

    fn log2_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::log2_(self, out)?.into())
    }

    fn log10(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::log10(self)?.into())
    }

    fn log10_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::log10_(self, out)?.into())
    }

    fn celu(&self, alpha: Self::OutputMeta) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::celu(self, alpha)?.into())
    }

    fn celu_<U>(&self, alpha: Self::OutputMeta, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::celu_(self, alpha, out)?.into())
    }

    fn sigmoid(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::sigmoid(self)?.into())
    }

    fn sigmoid_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::sigmoid_(self, out)?.into())
    }

    fn elu(&self, alpha: Self::OutputMeta) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::elu(self, alpha)?.into())
    }

    fn elu_<U>(&self, alpha: Self::OutputMeta, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::elu_(self, alpha, out)?.into())
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::leaky_relu(self, alpha)?.into())
    }

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::leaky_relu_(self, alpha, out)?.into())
    }

    fn gelu(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::gelu(self)?.into())
    }

    fn gelu_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::gelu_(self, out)?.into())
    }

    fn selu(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
    ) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::selu(self, alpha, gamma)?.into())
    }

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U,
    ) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::selu_(self, alpha, gamma, out)?.into())
    }

    fn hard_sigmoid(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::hard_sigmoid(self)?.into())
    }

    fn hard_sigmoid_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::hard_sigmoid_(self, out)?.into())
    }

    fn hard_swish(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::hard_swish(self)?.into())
    }

    fn hard_swish_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::hard_swish_(self, out)?.into())
    }

    fn relu6(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::relu6(self)?.into())
    }

    fn relu6_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::relu6_(self, out)?.into())
    }

    fn softplus(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::softplus(self)?.into())
    }

    fn softplus_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::softplus_(self, out)?.into())
    }

    fn softsign(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::softsign(self)?.into())
    }

    fn softsign_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::softsign_(self, out)?.into())
    }

    fn mish(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::mish(self)?.into())
    }

    fn mish_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu>::mish_(self, out)?.into())
    }
}
