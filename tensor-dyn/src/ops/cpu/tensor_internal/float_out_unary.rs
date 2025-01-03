use std::borrow::Borrow;

use tensor_common::error::base::TensorError;
use tensor_traits::{CommonBounds, FloatUaryOps};
use tensor_types::traits::VecTrait;
use tensor_types::{
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{FloatOutBinary, FloatOutUnary},
};

use crate::{ops::cpu::unary::uary_fn_with_out_simd, tensor_base::_Tensor};

pub(crate) type FloatUnaryType<T> = <T as FloatOutUnary>::Output;
pub(crate) type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T> FloatUaryOps for _Tensor<T>
where
    T: FloatOutUnary + CommonBounds,
    FloatUnaryType<T>: CommonBounds,
    f64: IntoScalar<<T as FloatOutUnary>::Output>,
    T::Vec: FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec>,
{
    type Output = _Tensor<FloatUnaryType<T>>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>>;

    type OutputMeta = FloatUnaryType<T>;

    fn sin(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._sin(),
            |x| x._sin(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn cos(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._cos(),
            |x| x._cos(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn tan(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._tan(),
            |x| x._tan(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn asin(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._asin(),
            |x| x._asin(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn acos(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._acos(),
            |x| x._acos(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn atan(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._atan(),
            |x| x._atan(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sinh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._sinh(),
            |x| x._sinh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn cosh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._cosh(),
            |x| x._cosh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn tanh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._tanh(),
            |x| x._tanh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn asinh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._asinh(),
            |x| x._asinh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn acosh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._acosh(),
            |x| x._acosh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn atanh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._atanh(),
            |x| x._atanh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<_Tensor<FloatUnaryType<T>>>,
    {
        uary_fn_with_out_simd(self, |x| x._sin(), |x| x._sin(), Some(out))
    }

    fn cos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._cos(), |x| x._cos(), Some(out))
    }

    fn tan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._tan(), |x| x._tan(), Some(out))
    }

    fn asin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._asin(), |x| x._asin(), Some(out))
    }

    fn acos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._acos(), |x| x._acos(), Some(out))
    }

    fn atan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._atan(), |x| x._atan(), Some(out))
    }

    fn sinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sinh(), |x| x._sinh(), Some(out))
    }

    fn cosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._cosh(), |x| x._cosh(), Some(out))
    }

    fn tanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._tanh(), |x| x._tanh(), Some(out))
    }

    fn asinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._asinh(), |x| x._asinh(), Some(out))
    }

    fn acosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._acosh(), |x| x._acosh(), Some(out))
    }

    fn atanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._atanh(), |x| x._atanh(), Some(out))
    }

    fn exp(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._exp(),
            |x| x._exp(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn exp_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._exp(), |x| x._exp(), Some(out))
    }

    fn exp2(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._exp2(),
            |x| x._exp2(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn exp2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._exp2(), |x| x._exp2(), Some(out))
    }

    fn sqrt(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._sqrt(),
            |x| x._sqrt(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sqrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sqrt(), |x| x._sqrt(), Some(out))
    }

    fn recip(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._recip(),
            |x| x._recip(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn recip_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._recip(), |x| x._recip(), Some(out))
    }

    fn ln(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._ln(),
            |x| x._ln(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn ln_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._ln(), |x| x._ln(), Some(out))
    }

    fn log2(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._log2(),
            |x| x._log2(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn log2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._log2(), |x| x._log2(), Some(out))
    }

    fn log10(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._log10(),
            |x| x._log10(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn log10_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._log10(), |x| x._log10(), Some(out))
    }

    fn celu(&self, alpha: FloatUnaryType<T>) -> std::result::Result<Self::Output, TensorError> {
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        uary_fn_with_out_simd(
            self,
            move |x| x._celu(alpha_vec),
            move |x| x._celu(alpha),
            None::<_Tensor<FloatUnaryType<T>>>,
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
        uary_fn_with_out_simd(self, |x| x._celu(alpha_vec), |x| x._celu(alpha), Some(out))
    }

    fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._sigmoid(),
            |x| x._sigmoid(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sigmoid(), |x| x._sigmoid(), Some(out))
    }

    fn elu(&self, alpha: FloatUnaryType<T>) -> std::result::Result<Self::Output, TensorError> {
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        uary_fn_with_out_simd(
            self,
            |x| x._elu(alpha_vec),
            |x| x._elu(alpha),
            None::<_Tensor<FloatUnaryType<T>>>,
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
        uary_fn_with_out_simd(self, |x| x._elu(alpha_vec), |x| x._elu(alpha), Some(out))
    }

    fn erf(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._erf(),
            |x| x._erf(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn fast_hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._fast_hard_sigmoid(),
            |x| x._fast_hard_sigmoid(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn gelu(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._gelu(),
            |x| x._gelu(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn gelu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._gelu(), |x| x._gelu(), Some(out))
    }

    fn selu<U>(&self, alpha: U, gamma: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Into<Option<Self::OutputMeta>>,
    {
        let alpha = alpha.into();
        let gamma = gamma.into();
        let alpha = alpha.unwrap_or((1.6732632423543772848170429916717).into_scalar());
        let gamma = gamma.unwrap_or((1.0507009873554804934193349852946).into_scalar());
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        let gamma_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(gamma);
        uary_fn_with_out_simd(
            self,
            |x| x._selu(alpha_vec, gamma_vec),
            |x| x._selu(alpha, gamma),
            None::<_Tensor<FloatUnaryType<T>>>,
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
        let alpha = alpha.unwrap_or((1.6732632423543772848170429916717).into_scalar());
        let gamma = gamma.unwrap_or((1.0507009873554804934193349852946).into_scalar());
        let alpha_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(alpha);
        let gamma_vec = <FloatUnaryType<T> as TypeCommon>::Vec::splat(gamma);
        uary_fn_with_out_simd(
            self,
            |x| x._selu(alpha_vec, gamma_vec),
            |x| x._selu(alpha, gamma),
            Some(out),
        )
    }

    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._hard_sigmoid(),
            |x| x._hard_sigmoid(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn hard_sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            |x| x._hard_sigmoid(),
            |x| x._hard_sigmoid(),
            Some(out),
        )
    }

    fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._hard_swish(),
            |x| x._hard_swish(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn hard_swish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._hard_swish(), |x| x._hard_swish(), Some(out))
    }

    fn softplus(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._softplus(),
            |x| x._softplus(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn softplus_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._softplus(), |x| x._softplus(), Some(out))
    }

    fn softsign(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._softsign(),
            |x| x._softsign(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn softsign_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._softsign(), |x| x._softsign(), Some(out))
    }

    fn mish(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            |x| x._mish(),
            |x| x._mish(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn mish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._mish(), |x| x._mish(), Some(out))
    }

    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(self, |x| x._cbrt(), |x| x._cbrt(), None::<Self::Output>)
    }

    fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._cbrt(), |x| x._cbrt(), Some(out))
    }
}
