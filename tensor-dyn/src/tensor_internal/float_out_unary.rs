use std::borrow::Borrow;

use tensor_traits::{CommonBounds, FloatUaryOps};
use tensor_types::{dtype::TypeCommon, into_scalar::IntoScalar, type_promote::{FloatOutBinary, FloatOutUnary}};

use crate::{ops::cpu::unary::uary_fn_with_out_simd, tensor_base::_Tensor};

pub(crate) type FloatUnaryType<T> = <T as FloatOutUnary>::Output;
pub(crate) type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T> FloatUaryOps for _Tensor<T>
where
    T: FloatOutUnary<Base = FloatUnaryType<T>> + CommonBounds,
    FloatUnaryType<T>: CommonBounds,
    f64: IntoScalar<<T as FloatOutUnary>::Output>,
    T::Vec:
        FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec, Base = FloatUnaryType<T>>,
{
    type Output = _Tensor<FloatUnaryType<T>>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>>;

    type OutputMeta = FloatUnaryType<T>;

    fn sin(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sin(),
            |x| x._sin(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn cos(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._cos(),
            |x| x._cos(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn tan(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._tan(),
            |x| x._tan(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn asin(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._asin(),
            |x| x._asin(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn acos(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._acos(),
            |x| x._acos(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn atan(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._atan(),
            |x| x._atan(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sinh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sinh(),
            |x| x._sinh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn cosh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._cosh(),
            |x| x._cosh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn erf(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._erf(),
            |x| x._erf(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn tanh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._tanh(),
            |x| x._tanh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn asinh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._asinh(),
            |x| x._asinh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn acosh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._acosh(),
            |x| x._acosh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn atanh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._atanh(),
            |x| x._atanh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sin_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<_Tensor<FloatUnaryType<T>>>,
    {
        uary_fn_with_out_simd(self, |x| x._sin(), |x| x._sin(), Some(out))
    }

    fn cos_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._cos(), |x| x._cos(), Some(out))
    }

    fn tan_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._tan(), |x| x._tan(), Some(out))
    }

    fn asin_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._asin(), |x| x._asin(), Some(out))
    }

    fn acos_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._acos(), |x| x._acos(), Some(out))
    }

    fn atan_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._atan(), |x| x._atan(), Some(out))
    }

    fn sinh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sinh(), |x| x._sinh(), Some(out))
    }

    fn cosh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._cosh(), |x| x._cosh(), Some(out))
    }

    fn tanh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._tanh(), |x| x._tanh(), Some(out))
    }

    fn asinh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._asinh(), |x| x._asinh(), Some(out))
    }

    fn acosh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._acosh(), |x| x._acosh(), Some(out))
    }

    fn atanh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._atanh(), |x| x._atanh(), Some(out))
    }

    fn exp(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._exp(),
            |x| x._exp(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn exp_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._exp(), |x| x._exp(), Some(out))
    }

    fn exp2(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._exp2(),
            |x| x._exp2(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn exp2_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._exp2(), |x| x._exp2(), Some(out))
    }

    fn sqrt(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sqrt(),
            |x| x._sqrt(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sqrt_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sqrt(), |x| x._sqrt(), Some(out))
    }

    fn recip(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._recip(),
            |x| x._recip(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn recip_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._recip(), |x| x._recip(), Some(out))
    }

    fn ln(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._ln(),
            |x| x._ln(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn ln_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._ln(), |x| x._ln(), Some(out))
    }

    fn log2(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._log2(),
            |x| x._log2(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn log2_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._log2(), |x| x._log2(), Some(out))
    }

    fn log10(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._log10(),
            |x| x._log10(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn log10_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._log10(), |x| x._log10(), Some(out))
    }

    fn celu(&self, alpha: FloatUnaryType<T>) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._celu(alpha),
            |x| x._celu(alpha),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn celu_<U>(
        &self,
        alpha: FloatUnaryType<T>,
        out: U,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._celu(alpha), |x| x._celu(alpha), Some(out))
    }

    fn sigmoid(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sigmoid(),
            |x| x._sigmoid(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sigmoid_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sigmoid(), |x| x._sigmoid(), Some(out))
    }

    fn elu(&self, alpha: FloatUnaryType<T>) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._elu(alpha),
            |x| x._elu(alpha),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn elu_<U>(
        &self,
        alpha: FloatUnaryType<T>,
        out: U,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._elu(alpha), |x| x._elu(alpha), Some(out))
    }

    fn leaky_relu(&self, alpha: FloatUnaryType<T>) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn leaky_relu_<U>(
        &self,
        alpha: FloatUnaryType<T>,
        out: U,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha),
            Some(out),
        )
    }

    fn gelu(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._gelu(),
            |x| x._gelu(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn gelu_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._gelu(), |x| x._gelu(), Some(out))
    }

    fn selu(
        &self,
        alpha: Option<FloatUnaryType<T>>,
        gamma: Option<FloatUnaryType<T>>,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        let alpha = alpha.unwrap_or(1.6732632423543772848170429916717.into_scalar());
        let gamma = gamma.unwrap_or(1.0507009873554804934193349852946.into_scalar());
        uary_fn_with_out_simd(
            self,
            |x| x._selu(alpha, gamma),
            |x| x._selu(alpha, gamma),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn selu_<U>(
        &self,
        alpha: Option<FloatUnaryType<T>>,
        gamma: Option<FloatUnaryType<T>>,
        out: U,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        let alpha = alpha.unwrap_or(1.67326319217681884765625.into_scalar());
        let gamma = gamma.unwrap_or(1.05070102214813232421875.into_scalar());
        uary_fn_with_out_simd(
            self,
            |x| x._selu(alpha, gamma),
            |x| x._selu(alpha, gamma),
            Some(out),
        )
    }

    fn hard_sigmoid(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._hard_sigmoid(),
            |x| x._hard_sigmoid(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn fast_hard_sigmoid(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._fast_hard_sigmoid(),
            |x| x._fast_hard_sigmoid(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn hard_sigmoid_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
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

    fn hard_swish(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._hard_swish(),
            |x| x._hard_swish(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn hard_swish_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._hard_swish(), |x| x._hard_swish(), Some(out))
    }

    fn relu6(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu6(),
            |x| x._relu6(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn relu6_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._relu6(), |x| x._relu6(), Some(out))
    }

    fn softplus(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._softplus(),
            |x| x._softplus(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn softplus_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._softplus(), |x| x._softplus(), Some(out))
    }

    fn softsign(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._softsign(),
            |x| x._softsign(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn softsign_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._softsign(), |x| x._softsign(), Some(out))
    }

    fn mish(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._mish(),
            |x| x._mish(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn mish_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._mish(), |x| x._mish(), Some(out))
    }

    fn relu(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu(),
            |x| x._relu(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn relu_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._relu(), |x| x._relu(), Some(out))
    }
}
