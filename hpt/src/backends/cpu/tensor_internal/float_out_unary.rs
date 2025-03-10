use std::borrow::BorrowMut;

use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_iterator::iterator_traits::{ParStridedIteratorSimd, ParStridedIteratorSimdZip};
use hpt_iterator::TensorIterator;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::unary::FloatUnaryOps;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::traits::VecTrait;
use hpt_types::{dtype::TypeCommon, into_scalar::Cast, type_promote::FloatOutUnary};

use crate::{backends::cpu::utils::unary::unary::unary_fn_with_out, tensor_base::_Tensor};

type FloatUnaryType<T> = <T as FloatOutUnary>::Output;

impl<T, A2, const DEVICE: usize> FloatUnaryOps for _Tensor<T, Cpu, DEVICE, A2>
where
    T: FloatOutUnary + CommonBounds,
    FloatUnaryType<T>: CommonBounds,
    f64: Cast<<T as FloatOutUnary>::Output>,
    T::Vec: FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec>,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<FloatUnaryType<T>, Cpu, DEVICE, A2>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>, Cpu, DEVICE, A2>;

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

    fn sincos(&self) -> std::result::Result<(Self::Output, Self::Output), TensorError> {
        let mut res1 = Self::InplaceOutput::empty(self.shape())?;
        let mut res2 = Self::InplaceOutput::empty(self.shape())?;
        res1.par_iter_mut_simd()
            .zip(res2.par_iter_mut_simd())
            .zip(self.par_iter_simd())
            .for_each(
                |((res1, res2), x)| {
                    let (sin, cos) = x._sincos();
                    *res1 = sin;
                    *res2 = cos;
                },
                |((a, b), x)| {
                    let (sin, cos) = x._sincos();
                    a.write_unaligned(sin);
                    b.write_unaligned(cos);
                },
            );
        Ok((res1, res2))
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._sin(), |x| x._sin(), Some(out))
    }

    fn cos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._cos(), |x| x._cos(), Some(out))
    }

    fn tan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._tan(), |x| x._tan(), Some(out))
    }

    fn asin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._asin(), |x| x._asin(), Some(out))
    }

    fn acos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._acos(), |x| x._acos(), Some(out))
    }

    fn atan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._atan(), |x| x._atan(), Some(out))
    }

    fn sinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._sinh(), |x| x._sinh(), Some(out))
    }

    fn cosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._cosh(), |x| x._cosh(), Some(out))
    }

    fn tanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._tanh(), |x| x._tanh(), Some(out))
    }

    fn asinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._asinh(), |x| x._asinh(), Some(out))
    }

    fn acosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._acosh(), |x| x._acosh(), Some(out))
    }

    fn atanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._recip(), |x| x._recip(), Some(out))
    }

    fn ln(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._ln(), |x| x._ln(), None::<Self::InplaceOutput>)
    }

    fn ln_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._log10(), |x| x._log10(), Some(out))
    }

    fn celu<V: Cast<Self::OutputMeta>>(
        &self,
        alpha: V,
    ) -> std::result::Result<Self::Output, TensorError> {
        let alpha: Self::OutputMeta = alpha.cast();
        let alpha_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(alpha);
        unary_fn_with_out(
            self,
            move |x| x._celu(alpha_vec),
            move |x| x._celu(alpha),
            None::<Self::InplaceOutput>,
        )
    }

    fn celu_<V, U>(&self, alpha: V, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
        V: Cast<Self::OutputMeta>,
    {
        let alpha: Self::OutputMeta = alpha.cast();
        let alpha_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(alpha);
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._sigmoid(), |x| x._sigmoid(), Some(out))
    }

    fn elu<V: Cast<Self::OutputMeta>>(
        &self,
        alpha: V,
    ) -> std::result::Result<Self::Output, TensorError> {
        let alpha: Self::OutputMeta = alpha.cast();
        let alpha_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(alpha);
        unary_fn_with_out(
            self,
            |x| x._elu(alpha_vec),
            |x| x._elu(alpha),
            None::<Self::InplaceOutput>,
        )
    }

    fn elu_<V, U>(&self, alpha: V, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
        V: Cast<Self::OutputMeta>,
    {
        let alpha: Self::OutputMeta = alpha.cast();
        let alpha_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(alpha);
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._gelu(), |x| x._gelu(), Some(out))
    }

    fn selu(&self) -> std::result::Result<Self::Output, TensorError> {
        let alpha: Self::OutputMeta = (1.6732632423543772848170429916717).cast();
        let gamma: Self::OutputMeta = (1.0507009873554804934193349852946).cast();
        let alpha_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(alpha);
        let gamma_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(gamma);
        unary_fn_with_out(
            self,
            |x| x._selu(alpha_vec, gamma_vec),
            |x| x._selu(alpha, gamma),
            None::<Self::InplaceOutput>,
        )
    }

    fn selu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        let alpha: Self::OutputMeta = (1.6732632423543772848170429916717).cast();
        let gamma: Self::OutputMeta = (1.0507009873554804934193349852946).cast();
        let alpha_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(alpha);
        let gamma_vec = <Self::OutputMeta as TypeCommon>::Vec::splat(gamma);
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
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
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._mish(), |x| x._mish(), Some(out))
    }

    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(self, |x| x._cbrt(), |x| x._cbrt(), None::<Self::Output>)
    }

    fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._cbrt(), |x| x._cbrt(), Some(out))
    }

    fn exp10(&self) -> std::result::Result<Self::Output, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._exp10(),
            |x| x._exp10(),
            None::<Self::InplaceOutput>,
        )
    }

    fn exp10_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._exp10(), |x| x._exp10(), Some(out))
    }

    fn sincos_<U, O>(
        &self,
        (mut out1, mut out2): (U, O),
    ) -> std::result::Result<(Self::Output, Self::Output), TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
        O: BorrowMut<Self::InplaceOutput>,
    {
        use hpt_common::error::shape::ShapeError;
        let res1: &mut Self::InplaceOutput = out1.borrow_mut();
        let res2: &mut Self::InplaceOutput = out2.borrow_mut();
        ShapeError::check_inplace_out_layout_valid(res1.shape(), res1.layout())?;
        ShapeError::check_inplace_out_layout_valid(res2.shape(), res2.layout())?;
        res1.par_iter_mut_simd()
            .zip(res2.par_iter_mut_simd())
            .zip(self.par_iter_simd())
            .for_each(
                |((res1, res2), x)| {
                    let (sin, cos) = x._sincos();
                    *res1 = sin;
                    *res2 = cos;
                },
                |((a, b), x)| {
                    let (sin, cos) = x._sincos();
                    a.write_unaligned(sin);
                    b.write_unaligned(cos);
                },
            );
        Ok((res1.clone(), res2.clone()))
    }

    fn erf_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        unary_fn_with_out(self, |x| x._erf(), |x| x._erf(), Some(out))
    }
}
