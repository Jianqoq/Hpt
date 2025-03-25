use std::borrow::BorrowMut;

use crate::{
    backend::Cuda,
    backends::cuda::utils::unary::unary::{uary_fn_precompiled, uary_fn_precompiled_1scalar},
    tensor_base::_Tensor,
};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{ops::unary::FloatUnaryOps, tensor::CommonBounds};
use hpt_types::{cuda_types::scalar::Scalar, dtype::CudaType};
use hpt_types::{into_scalar::Cast, type_promote::FloatOutUnary};

pub(crate) type FloatUnaryType<T> = <T as FloatOutUnary>::Output;
use hpt_cudakernels::*;

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
        uary_fn_precompiled(self, "sin", &SIN, None::<Self::InplaceOutput>)
    }

    fn cos(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "cos", &COS, None::<Self::InplaceOutput>)
    }

    fn tan(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "tan", &TAN, None::<Self::InplaceOutput>)
    }

    fn asin(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "asin", &ASIN, None::<Self::InplaceOutput>)
    }

    fn acos(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "acos", &ACOS, None::<Self::InplaceOutput>)
    }

    fn atan(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "atan", &ATAN, None::<Self::InplaceOutput>)
    }

    fn sinh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "sinh", &SINH, None::<Self::InplaceOutput>)
    }

    fn cosh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "cosh", &COSH, None::<Self::InplaceOutput>)
    }

    fn tanh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "tanh", &TANH, None::<Self::InplaceOutput>)
    }

    fn asinh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "asinh", &ASINH, None::<Self::InplaceOutput>)
    }

    fn acosh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "acosh", &ACOSH, None::<Self::InplaceOutput>)
    }

    fn atanh(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "atanh", &ATANH, None::<Self::InplaceOutput>)
    }

    fn sin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "sin", &SIN, Some(out))
    }

    fn cos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "cos", &COS, Some(out))
    }

    fn tan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "tan", &TAN, Some(out))
    }

    fn asin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "asin", &ASIN, Some(out))
    }

    fn acos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "acos", &ACOS, Some(out))
    }

    fn atan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "atan", &ATAN, Some(out))
    }

    fn sinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "sinh", &SINH, Some(out))
    }

    fn cosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "cosh", &COSH, Some(out))
    }

    fn tanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "tanh", &TANH, Some(out))
    }

    fn asinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "asinh", &ASINH, Some(out))
    }

    fn acosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "acosh", &ACOSH, Some(out))
    }

    fn atanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "atanh", &ATANH, Some(out))
    }

    fn exp(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "exp", &EXP, None::<Self::InplaceOutput>)
    }

    fn exp_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "exp", &EXP, Some(out))
    }

    fn exp2(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "exp2", &EXP2, None::<Self::InplaceOutput>)
    }

    fn exp2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "exp2", &EXP2, Some(out))
    }

    fn sqrt(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "sqrt", &SQRT, None::<Self::InplaceOutput>)
    }

    fn sqrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "sqrt", &SQRT, Some(out))
    }

    fn recip(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "recip", &RECIP, None::<Self::InplaceOutput>)
    }

    fn recip_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "recip", &RECIP, Some(out))
    }

    fn ln(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "ln", &LN, None::<Self::InplaceOutput>)
    }

    fn ln_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "ln", &LN, Some(out))
    }

    fn log2(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "log2", &LOG2, None::<Self::InplaceOutput>)
    }

    fn log2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "log2", &LOG2, Some(out))
    }

    fn log10(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "log10", &LOG10, None::<Self::InplaceOutput>)
    }

    fn log10_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "log10", &LOG10, Some(out))
    }

    fn celu<V: Cast<Self::OutputMeta>>(
        &self,
        alpha: V,
    ) -> std::result::Result<Self::Output, TensorError> {
        let alpha: Self::OutputMeta = alpha.cast();
        uary_fn_precompiled_1scalar(self, "celu", &CELU, alpha, None::<Self::InplaceOutput>)
    }

    fn celu_<V, U>(&self, alpha: V, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        V: Cast<Self::OutputMeta>,
        U: BorrowMut<Self::InplaceOutput>,
    {
        let alpha: Self::OutputMeta = alpha.cast();
        uary_fn_precompiled_1scalar(self, "celu", &CELU, alpha, Some(out))
    }

    fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "sigmoid", &SIGMOID, None::<Self::InplaceOutput>)
    }

    fn sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "sigmoid", &SIGMOID, Some(out))
    }

    fn elu<V: Cast<Self::OutputMeta>>(
        &self,
        alpha: V,
    ) -> std::result::Result<Self::Output, TensorError> {
        let alpha: Self::OutputMeta = alpha.cast();
        uary_fn_precompiled_1scalar(self, "elu", &ELU, alpha, None::<Self::InplaceOutput>)
    }

    fn elu_<V, U>(&self, alpha: V, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        V: Cast<Self::OutputMeta>,
        U: BorrowMut<Self::InplaceOutput>,
    {
        let alpha: Self::OutputMeta = alpha.cast();
        uary_fn_precompiled_1scalar(self, "elu", &ELU, alpha, Some(out))
    }

    fn erf(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "erf", &ERF, None::<Self::InplaceOutput>)
    }

    fn gelu(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "gelu", &GELU, None::<Self::InplaceOutput>)
    }

    fn gelu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "gelu", &GELU, Some(out))
    }

    fn selu(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "selu", &SELU, None::<Self::InplaceOutput>)
    }

    fn selu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "selu", &SELU, Some(out))
    }

    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(
            self,
            "hard_sigmoid",
            &HARD_SIGMOID,
            None::<Self::InplaceOutput>,
        )
    }

    fn hard_sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "hard_sigmoid", &HARD_SIGMOID, Some(out))
    }

    fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "hard_swish", &HARD_SWISH, None::<Self::InplaceOutput>)
    }

    fn hard_swish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "hard_swish", &HARD_SWISH, Some(out))
    }

    fn softplus(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "soft_plus", &SOFTPLUS, None::<Self::InplaceOutput>)
    }

    fn softplus_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "soft_plus", &SOFTPLUS, Some(out))
    }

    fn softsign(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "soft_sign", &SOFTSIGN, None::<Self::InplaceOutput>)
    }

    fn softsign_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "soft_sign", &SOFTSIGN, Some(out))
    }

    fn mish(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "mish", &MISH, None::<Self::InplaceOutput>)
    }

    fn mish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "mish", &MISH, Some(out))
    }

    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError> {
        uary_fn_precompiled(self, "cbrt", &CBRT, None::<Self::InplaceOutput>)
    }

    fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "cbrt", &CBRT, Some(out))
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
        uary_fn_precompiled(self, "exp10", &EXP10, None::<Self::InplaceOutput>)
    }

    fn exp10_<U>(
        &self,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, hpt_common::error::base::TensorError>
    where
        U: std::borrow::BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "exp10", &EXP10, Some(out))
    }

    fn erf_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        uary_fn_precompiled(self, "erf", &ERF, Some(out))
    }
}
