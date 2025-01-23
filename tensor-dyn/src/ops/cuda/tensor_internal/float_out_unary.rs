use std::borrow::Borrow;

use crate::{
    ops::cuda::{cuda_utils::get_module_name_1, unary::uary_fn_with_out_simd},
    tensor_base::_Tensor,
    Cuda,
};
use cudarc::driver::DeviceRepr;
use tensor_common::err_handler::TensorError;
use tensor_traits::{CommonBounds, FloatUnaryOps};
use tensor_types::cuda_types::scalar::Scalar;
use tensor_types::{dtype::TypeCommon, into_scalar::IntoScalar, type_promote::FloatOutUnary};

pub(crate) type FloatUnaryType<T> = <T as FloatOutUnary>::Output;

impl<T, const DEVICE_ID: usize> FloatUnaryOps for _Tensor<T, Cuda, DEVICE_ID>
where
    T: FloatOutUnary<Base = FloatUnaryType<T>> + CommonBounds + DeviceRepr,
    FloatUnaryType<T>: CommonBounds + DeviceRepr,
    f64: IntoScalar<<T as FloatOutUnary>::Output>,
    T::Vec:
        FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec, Base = FloatUnaryType<T>>,
    Scalar<T>: FloatOutUnary<Output = Scalar<FloatUnaryType<T>>, Base = Scalar<FloatUnaryType<T>>>,
{
    type Output = _Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID>;

    type OutputMeta = FloatUnaryType<T>;

    fn sin(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sin", self),
            |out, x| out.assign(x._sin()),
            None::<Self::InplaceOutput>,
        )
    }

    fn cos(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cos", self),
            |out, x| out.assign(x._cos()),
            None::<Self::InplaceOutput>,
        )
    }

    fn tan(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("tan", self),
            |out, x| out.assign(x._tan()),
            None::<Self::InplaceOutput>,
        )
    }

    fn asin(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("asin", self),
            |out, x| out.assign(x._asin()),
            None::<Self::InplaceOutput>,
        )
    }

    fn acos(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("acos", self),
            |out, x| out.assign(x._acos()),
            None::<Self::InplaceOutput>,
        )
    }

    fn atan(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("atan", self),
            |out, x| out.assign(x._atan()),
            None::<Self::InplaceOutput>,
        )
    }

    fn sinh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sinh", self),
            |out, x| out.assign(x._sinh()),
            None::<Self::InplaceOutput>,
        )
    }

    fn cosh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cosh", self),
            |out, x| out.assign(x._cosh()),
            None::<Self::InplaceOutput>,
        )
    }

    fn tanh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("tanh", self),
            |out, x| out.assign(x._tanh()),
            None::<Self::InplaceOutput>,
        )
    }

    fn asinh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("asinh", self),
            |out, x| out.assign(x._asinh()),
            None::<Self::InplaceOutput>,
        )
    }

    fn acosh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("acosh", self),
            |out, x| out.assign(x._acosh()),
            None::<Self::InplaceOutput>,
        )
    }

    fn atanh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("atanh", self),
            |out, x| out.assign(x._atanh()),
            None::<Self::InplaceOutput>,
        )
    }

    fn sin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<_Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID>>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sin", self),
            |out, x| out.assign(x._sin()),
            Some(out),
        )
    }

    fn cos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cos", self),
            |out, x| out.assign(x._cos()),
            Some(out),
        )
    }

    fn tan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("tan", self),
            |out, x| out.assign(x._tan()),
            Some(out),
        )
    }

    fn asin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("asin", self),
            |out, x| out.assign(x._asin()),
            Some(out),
        )
    }

    fn acos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("acos", self),
            |out, x| out.assign(x._acos()),
            Some(out),
        )
    }

    fn atan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("atan", self),
            |out, x| out.assign(x._atan()),
            Some(out),
        )
    }

    fn sinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sinh", self),
            |out, x| out.assign(x._sinh()),
            Some(out),
        )
    }

    fn cosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cosh", self),
            |out, x| out.assign(x._cosh()),
            Some(out),
        )
    }

    fn tanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("tanh", self),
            |out, x| out.assign(x._tanh()),
            Some(out),
        )
    }

    fn asinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("asinh", self),
            |out, x| out.assign(x._asinh()),
            Some(out),
        )
    }

    fn acosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("acosh", self),
            |out, x| out.assign(x._acosh()),
            Some(out),
        )
    }

    fn atanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("atanh", self),
            |out, x| out.assign(x._atanh()),
            Some(out),
        )
    }

    fn exp(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp", self),
            |out, x| out.assign(x._exp()),
            None::<Self::InplaceOutput>,
        )
    }

    fn exp_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp", self),
            |out, x| out.assign(x._exp()),
            Some(out),
        )
    }

    fn exp2(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp2", self),
            |out, x| out.assign(x._exp2()),
            None::<Self::InplaceOutput>,
        )
    }

    fn exp2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp2", self),
            |out, x| out.assign(x._exp2()),
            Some(out),
        )
    }

    fn sqrt(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sqrt", self),
            |out, x| out.assign(x._sqrt()),
            None::<Self::InplaceOutput>,
        )
    }

    fn sqrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sqrt", self),
            |out, x| out.assign(x._sqrt()),
            Some(out),
        )
    }

    fn recip(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("recip", self),
            |out, x| out.assign(x._recip()),
            None::<Self::InplaceOutput>,
        )
    }

    fn recip_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("recip", self),
            |out, x| out.assign(x._recip()),
            Some(out),
        )
    }

    fn ln(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ln", self),
            |out, x| out.assign(x._ln()),
            None::<Self::InplaceOutput>,
        )
    }

    fn ln_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("ln", self),
            |out, x| out.assign(x._ln()),
            Some(out),
        )
    }

    fn log2(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("log2", self),
            |out, x| out.assign(x._log2()),
            None::<Self::InplaceOutput>,
        )
    }

    fn log2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("log2", self),
            |out, x| out.assign(x._log2()),
            Some(out),
        )
    }

    fn log10(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("log10", self),
            |out, x| out.assign(x._log10()),
            None::<Self::InplaceOutput>,
        )
    }

    fn log10_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("log10", self),
            |out, x| out.assign(x._log10()),
            Some(out),
        )
    }

    fn celu(&self, alpha: FloatUnaryType<T>) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("celu", self),
            |out, x| {
                let alpha = Scalar::new(format!("{}", alpha));
                out.assign(x._celu(alpha))
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn celu_<U>(&self, alpha: FloatUnaryType<T>, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("celu", self),
            |out, x| {
                let alpha = Scalar::new(format!("{}", alpha));
                out.assign(x._celu(alpha))
            },
            Some(out),
        )
    }

    fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sigmoid", self),
            |out, x| out.assign(x._sigmoid()),
            None::<Self::InplaceOutput>,
        )
    }

    fn sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sigmoid", self),
            |out, x| out.assign(x._sigmoid()),
            Some(out),
        )
    }

    fn elu(&self, alpha: FloatUnaryType<T>) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("elu", self),
            |out, x| {
                let alpha = Scalar::new(format!("{}", alpha));
                out.assign(x._elu(alpha))
            },
            None::<Self::InplaceOutput>,
        )
    }

    fn elu_<U>(&self, alpha: FloatUnaryType<T>, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("elu", self),
            |out, x| {
                let alpha = Scalar::new(format!("{}", alpha));
                out.assign(x._elu(alpha))
            },
            Some(out),
        )
    }

    fn erf(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("erf", self),
            |out, x| out.assign(x._erf()),
            None::<Self::InplaceOutput>,
        )
    }

    fn fast_hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("fast_hard_sigmoid", self),
            |out, x| out.assign(x._hard_sigmoid()),
            None::<Self::InplaceOutput>,
        )
    }

    fn gelu(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("gelu", self),
            |out, x| out.assign(x._gelu()),
            None::<Self::InplaceOutput>,
        )
    }

    fn gelu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("gelu", self),
            |out, x| out.assign(x._gelu()),
            Some(out),
        )
    }

    fn selu<U>(&self, alpha: U, gamma: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Into<Option<Self::OutputMeta>>,
    {
        let alpha = alpha.into();
        let gamma = gamma.into();
        let alpha = Scalar::new(format!(
            "{}",
            alpha.unwrap_or((1.6732632423543772848170429916717).into_scalar())
        ));
        let gamma = Scalar::new(format!(
            "{}",
            gamma.unwrap_or((1.0507009873554804934193349852946).into_scalar())
        ));
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("selu", self),
            |out, x| out.assign(x._selu(alpha.clone(), gamma.clone())),
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
        let alpha = Scalar::new(format!(
            "{}",
            alpha.unwrap_or((1.6732632423543772848170429916717).into_scalar())
        ));
        let gamma = Scalar::new(format!(
            "{}",
            gamma.unwrap_or((1.0507009873554804934193349852946).into_scalar())
        ));
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("selu", self),
            |out, x| out.assign(x._selu(alpha.clone(), gamma.clone())),
            Some(out),
        )
    }

    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("hard_sigmoid", self),
            |out, x| out.assign(x._hard_sigmoid()),
            None::<Self::InplaceOutput>,
        )
    }

    fn hard_sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("hard_sigmoid", self),
            |out, x| out.assign(x._hard_sigmoid()),
            Some(out),
        )
    }

    fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("hard_swish", self),
            |out, x| out.assign(x._hard_swish()),
            None::<Self::InplaceOutput>,
        )
    }

    fn hard_swish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("hard_swish", self),
            |out, x| out.assign(x._hard_swish()),
            Some(out),
        )
    }

    fn softplus(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("softplus", self),
            |out, x| out.assign(x._softplus()),
            None::<Self::InplaceOutput>,
        )
    }

    fn softplus_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("softplus", self),
            |out, x| out.assign(x._softplus()),
            Some(out),
        )
    }

    fn softsign(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("softsign", self),
            |out, x| out.assign(x._softsign()),
            None::<Self::InplaceOutput>,
        )
    }

    fn softsign_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("softsign", self),
            |out, x| out.assign(x._softsign()),
            Some(out),
        )
    }

    fn mish(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("mish", self),
            |out, x| out.assign(x._mish()),
            None::<Self::InplaceOutput>,
        )
    }

    fn mish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("mish", self),
            |out, x| out.assign(x._mish()),
            Some(out),
        )
    }
    
    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cbrt", self),
            |out, x| out.assign(x._cbrt()),
            None::<Self::InplaceOutput>,
        )
    }
    
    fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cbrt", self),
            |out, x| out.assign(x._cbrt()),
            Some(out),
        )
    }
}
