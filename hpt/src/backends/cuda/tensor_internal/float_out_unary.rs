use std::borrow::BorrowMut;

use crate::{
    backend::Cuda,
    backends::cuda::{cuda_utils::get_module_name_1, utils::unary::unary::uary_fn_with_out_simd},
    tensor_base::_Tensor,
};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{ops::unary::FloatUnaryOps, tensor::CommonBounds};
use hpt_types::{cuda_types::scalar::Scalar, dtype::CudaType};
use hpt_types::{into_scalar::Cast, type_promote::FloatOutUnary};

pub(crate) type FloatUnaryType<T> = <T as FloatOutUnary>::Output;

impl<T, const DEVICE_ID: usize, Al> FloatUnaryOps for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: FloatOutUnary + CommonBounds + DeviceRepr + CudaType,
    FloatUnaryType<T>: CommonBounds + DeviceRepr + CudaType,
    f64: Cast<<T as FloatOutUnary>::Output>,
    Scalar<T>: FloatOutUnary<Output = Scalar<FloatUnaryType<T>>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID, Al>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID, Al>;

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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("log10", self),
            |out, x| out.assign(x._log10()),
            Some(out),
        )
    }

    fn celu<V: Cast<Self::OutputMeta>>(
        &self,
        alpha: V,
    ) -> std::result::Result<Self::Output, TensorError> {
        let alpha: Self::OutputMeta = alpha.cast();
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

    fn celu_<V, U>(&self, alpha: V, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        V: Cast<Self::OutputMeta>,
        U: BorrowMut<Self::InplaceOutput>,
    {
        let alpha: Self::OutputMeta = alpha.cast();
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("sigmoid", self),
            |out, x| out.assign(x._sigmoid()),
            Some(out),
        )
    }

    fn elu<V: Cast<Self::OutputMeta>>(
        &self,
        alpha: V,
    ) -> std::result::Result<Self::Output, TensorError> {
        let alpha: Self::OutputMeta = alpha.cast();
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

    fn elu_<V, U>(&self, alpha: V, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        V: Cast<Self::OutputMeta>,
        U: BorrowMut<Self::InplaceOutput>,
    {
        let alpha: Self::OutputMeta = alpha.cast();
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("gelu", self),
            |out, x| out.assign(x._gelu()),
            Some(out),
        )
    }

    fn selu(&self) -> std::result::Result<Self::Output, TensorError> {
        let alpha: Self::OutputMeta = (1.6732632423543772848170429916717).cast();
        let gamma: Self::OutputMeta = (1.0507009873554804934193349852946).cast();
        let alpha = Scalar::new(format!("{}", alpha));
        let gamma = Scalar::new(format!("{}", gamma));
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("selu", self),
            |out, x| out.assign(x._selu(alpha.clone(), gamma.clone())),
            None::<Self::InplaceOutput>,
        )
    }

    fn selu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        let alpha: Self::OutputMeta = (1.6732632423543772848170429916717).cast();
        let gamma: Self::OutputMeta = (1.0507009873554804934193349852946).cast();
        let alpha = Scalar::new(format!("{}", alpha));
        let gamma = Scalar::new(format!("{}", gamma));
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("cbrt", self),
            |out, x| out.assign(x._cbrt()),
            Some(out),
        )
    }

    fn sincos(
        &self,
    ) -> std::result::Result<(Self::Output, Self::Output), hpt_common::error::base::TensorError>
    {
        let sin = self.sin()?;
        let cos = self.cos()?;
        Ok((sin, cos))
    }

    fn sincos_<U, O>(
        &self,
        (out1, out2): (U, O),
    ) -> std::result::Result<(Self::Output, Self::Output), hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
        O: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        let sin = self.sin_(out1)?;
        let cos = self.cos_(out2)?;
        Ok((sin, cos))
    }

    fn exp10(&self) -> std::result::Result<Self::Output, hpt_common::error::base::TensorError> {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp10", self),
            |out, x| out.assign(x._exp10()),
            None::<Self::InplaceOutput>,
        )
    }

    fn exp10_<U>(
        &self,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("exp10", self),
            |out, x| out.assign(x._exp10()),
            Some(out),
        )
    }

    fn erf_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("erf", self),
            |out, x| out.assign(x._erf()),
            Some(out),
        )
    }
}
