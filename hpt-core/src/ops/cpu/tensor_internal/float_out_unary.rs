use std::borrow::Borrow;

use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, FloatUnaryOps};
use hpt_types::traits::VecTrait;
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    type_promote::{FloatOutBinary, FloatOutUnary},
};

use crate::Cpu;
use crate::{ops::cpu::utils::unary::unary::unary_fn_with_out, tensor_base::_Tensor};

pub(crate) type FloatUnaryType<T> = <T as FloatOutUnary>::Output;
pub(crate) type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE: usize> FloatUnaryOps for _Tensor<T, Cpu, DEVICE>
where
    T: FloatOutUnary + CommonBounds,
    FloatUnaryType<T>: CommonBounds,
    f64: Cast<<T as FloatOutUnary>::Output>,
    T::Vec: FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec>,
{
    type Output = _Tensor<FloatUnaryType<T>, Cpu, DEVICE>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>, Cpu, DEVICE>;

    type OutputMeta = FloatUnaryType<T>;

    fn sin(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._sin(),
            |x| x._sin(),
            None::<Self::InplaceOutput>,
        )
    }

    fn cos(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._cos(),
            |x| x._cos(),
            None::<Self::InplaceOutput>,
        )
    }

    fn tan(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._tan(),
            |x| x._tan(),
            None::<Self::InplaceOutput>,
        )
    }

    fn asin(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._asin(),
            |x| x._asin(),
            None::<Self::InplaceOutput>,
        )
    }

    fn acos(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._acos(),
            |x| x._acos(),
            None::<Self::InplaceOutput>,
        )
    }

    fn atan(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._atan(),
            |x| x._atan(),
            None::<Self::InplaceOutput>,
        )
    }

    fn sinh(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._sinh(),
            |x| x._sinh(),
            None::<Self::InplaceOutput>,
        )
    }

    fn cosh(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._cosh(),
            |x| x._cosh(),
            None::<Self::InplaceOutput>,
        )
    }

    fn tanh(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._tanh(),
            |x| x._tanh(),
            None::<Self::InplaceOutput>,
        )
    }

    fn asinh(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._asinh(),
            |x| x._asinh(),
            None::<Self::InplaceOutput>,
        )
    }

    fn acosh(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._acosh(),
            |x| x._acosh(),
            None::<Self::InplaceOutput>,
        )
    }

    fn atanh(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._atanh(),
            |x| x._atanh(),
            None::<Self::InplaceOutput>,
        )
    }

    fn sin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._sin(), |x| x._sin(), Some(out))
    }

    fn cos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._cos(), |x| x._cos(), Some(out))
    }

    fn tan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._tan(), |x| x._tan(), Some(out))
    }

    fn asin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._asin(), |x| x._asin(), Some(out))
    }

    fn acos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._acos(), |x| x._acos(), Some(out))
    }

    fn atan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._atan(), |x| x._atan(), Some(out))
    }

    fn sinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._sinh(), |x| x._sinh(), Some(out))
    }

    fn cosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._cosh(), |x| x._cosh(), Some(out))
    }

    fn tanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._tanh(), |x| x._tanh(), Some(out))
    }

    fn asinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._asinh(), |x| x._asinh(), Some(out))
    }

    fn acosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._acosh(), |x| x._acosh(), Some(out))
    }

    fn atanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._atanh(), |x| x._atanh(), Some(out))
    }

    fn exp(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._exp(),
            |x| x._exp(),
            None::<Self::InplaceOutput>,
        )
    }

    fn exp_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._exp(), |x| x._exp(), Some(out))
    }

    fn exp2(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._exp2(),
            |x| x._exp2(),
            None::<Self::InplaceOutput>,
        )
    }

    fn exp2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._exp2(), |x| x._exp2(), Some(out))
    }

    fn sqrt(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._sqrt(),
            |x| x._sqrt(),
            None::<Self::InplaceOutput>,
        )
    }

    fn sqrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._sqrt(), |x| x._sqrt(), Some(out))
    }

    fn recip(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._recip(),
            |x| x._recip(),
            None::<Self::InplaceOutput>,
        )
    }

    fn recip_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._recip(), |x| x._recip(), Some(out))
    }

    fn ln(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._ln(), |x| x._ln(), None::<Self::InplaceOutput>)
    }

    fn ln_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._ln(), |x| x._ln(), Some(out))
    }

    fn log2(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._log2(),
            |x| x._log2(),
            None::<Self::InplaceOutput>,
        )
    }

    fn log2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._log2(), |x| x._log2(), Some(out))
    }

    fn log10(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._log10(),
            |x| x._log10(),
            None::<Self::InplaceOutput>,
        )
    }

    fn log10_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._log10(), |x| x._log10(), Some(out))
    }

    fn celu(&self, alpha: FloatUnaryType<T>) -> std::result::Result<Self::Output, TensorError> {
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        unary_fn_with_out(
            self,
            move |x| x._celu(alpha_vec),
            move |x| x._celu(alpha),
            None::<Self::InplaceOutput>,
        )
    }

    fn celu_<U>(
        &self,
        alpha: FloatUnaryType<T>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        unary_fn_with_out(self, |x| x._celu(alpha_vec), |x| x._celu(alpha), Some(out))
    }

    fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._sigmoid(),
            |x| x._sigmoid(),
            None::<Self::InplaceOutput>,
        )
    }

    fn sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._sigmoid(), |x| x._sigmoid(), Some(out))
    }

    fn elu(&self, alpha: FloatUnaryType<T>) -> std::result::Result<Self::Output, TensorError> {
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        unary_fn_with_out(
            self,
            |x| x._elu(alpha_vec),
            |x| x._elu(alpha),
            None::<Self::InplaceOutput>,
        )
    }

    fn elu_<U>(
        &self,
        alpha: FloatUnaryType<T>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        unary_fn_with_out(self, |x| x._elu(alpha_vec), |x| x._elu(alpha), Some(out))
    }

    fn erf(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._erf(),
            |x| x._erf(),
            None::<Self::InplaceOutput>,
        )
    }

    fn gelu(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._gelu(),
            |x| x._gelu(),
            None::<Self::InplaceOutput>,
        )
    }

    fn gelu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._gelu(), |x| x._gelu(), Some(out))
    }

    fn selu<U>(&self, alpha: U, gamma: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Into<Option<Self::OutputMeta>>,
    {
        let alpha = alpha.into();
        let gamma = gamma.into();
        let alpha = alpha.unwrap_or((1.6732632423543772848170429916717).cast());
        let gamma = gamma.unwrap_or((1.0507009873554804934193349852946).cast());
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        let gamma_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(gamma);
        unary_fn_with_out(
            self,
            |x| x._selu(alpha_vec, gamma_vec),
            |x| x._selu(alpha, gamma),
            None::<Self::InplaceOutput>,
        )
    }

    fn selu_<U>(
        &self,
        alpha: Option<FloatUnaryType<T>>,
        gamma: Option<FloatUnaryType<T>>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        let alpha = alpha.unwrap_or((1.6732632423543772848170429916717).cast());
        let gamma = gamma.unwrap_or((1.0507009873554804934193349852946).cast());
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        let gamma_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(gamma);
        unary_fn_with_out(
            self,
            |x| x._selu(alpha_vec, gamma_vec),
            |x| x._selu(alpha, gamma),
            Some(out),
        )
    }

    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._hard_sigmoid(),
            |x| x._hard_sigmoid(),
            None::<Self::InplaceOutput>,
        )
    }

    fn hard_sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(
            self,
            |x| x._hard_sigmoid(),
            |x| x._hard_sigmoid(),
            Some(out),
        )
    }

    fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._hard_swish(),
            |x| x._hard_swish(),
            None::<Self::InplaceOutput>,
        )
    }

    fn hard_swish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._hard_swish(), |x| x._hard_swish(), Some(out))
    }

    fn softplus(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._softplus(),
            |x| x._softplus(),
            None::<Self::InplaceOutput>,
        )
    }

    fn softplus_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._softplus(), |x| x._softplus(), Some(out))
    }

    fn softsign(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._softsign(),
            |x| x._softsign(),
            None::<Self::InplaceOutput>,
        )
    }

    fn softsign_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._softsign(), |x| x._softsign(), Some(out))
    }

    fn mish(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._mish(),
            |x| x._mish(),
            None::<Self::InplaceOutput>,
        )
    }

    fn mish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._mish(), |x| x._mish(), Some(out))
    }

    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._cbrt(), |x| x._cbrt(), None::<Self::Output>)
    }

    fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._cbrt(), |x| x._cbrt(), Some(out))
    }
}
