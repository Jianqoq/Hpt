use std::borrow::BorrowMut;

use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, FloatUnaryOps};
use hpt_types::{dtype::TypeCommon, into_scalar::Cast, type_promote::FloatOutUnary};

use crate::{
    ops::cpu::tensor_internal::float_out_unary::FloatUnaryType, tensor::Tensor,
    tensor_base::_Tensor, Cpu,
};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};

impl<T, const DEVICE: usize, Al> FloatUnaryOps for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds,
    FloatUnaryType<T>: CommonBounds,
    f64: Cast<<T as FloatOutUnary>::Output>,
    T::Vec: FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<FloatUnaryType<T>, Cpu, DEVICE, Al>;

    type InplaceOutput = Tensor<FloatUnaryType<T>, Cpu, DEVICE, Al>;

    type OutputMeta = FloatUnaryType<T>;

    fn sin(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sin(self.inner.as_ref())?.into())
    }

    fn cos(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::cos(self.inner.as_ref())?.into())
    }

    fn tan(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::tan(self.inner.as_ref())?.into())
    }

    fn asin(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::asin(self.inner.as_ref())?.into())
    }

    fn acos(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::acos(self.inner.as_ref())?.into())
    }

    fn atan(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::atan(self.inner.as_ref())?.into())
    }

    fn sinh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sinh(self.inner.as_ref())?.into())
    }

    fn cosh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::cosh(self.inner.as_ref())?.into())
    }

    fn tanh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::tanh(self.inner.as_ref())?.into())
    }

    fn asinh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::asinh(self.inner.as_ref())?.into())
    }

    fn acosh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::acosh(self.inner.as_ref())?.into())
    }

    fn atanh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::atanh(self.inner.as_ref())?.into())
    }

    fn sin_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sin_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn cos_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::cos_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn tan_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::tan_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn asin_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::asin_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn acos_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::acos_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn atan_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::atan_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sinh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sinh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn cosh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::cosh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn tanh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::tanh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn asinh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::asinh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn acosh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::acosh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn atanh_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::atanh_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn exp(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::exp(self.inner.as_ref())?.into())
    }

    fn exp_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::exp_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn exp2(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::exp2(self.inner.as_ref())?.into())
    }

    fn exp2_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::exp2_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sqrt(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sqrt(self.inner.as_ref())?.into())
    }

    fn sqrt_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sqrt_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn recip(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::recip(self.inner.as_ref())?.into())
    }

    fn recip_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::recip_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn ln(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::ln(self.inner.as_ref())?.into())
    }

    fn ln_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::ln_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn log2(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::log2(self.inner.as_ref())?.into())
    }

    fn log2_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::log2_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn log10(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::log10(self.inner.as_ref())?.into())
    }

    fn log10_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::log10_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn celu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::celu(self.inner.as_ref(), alpha)?.into())
    }

    fn celu_<U>(
        &self,
        alpha: Self::OutputMeta,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::celu_(
            self.inner.as_ref(),
            alpha,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sigmoid(self.inner.as_ref())?.into())
    }

    fn sigmoid_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::sigmoid_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn elu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::elu(self.inner.as_ref(), alpha)?.into())
    }

    fn elu_<U>(
        &self,
        alpha: Self::OutputMeta,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::elu_(
            self.inner.as_ref(),
            alpha,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn erf(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::erf(self.inner.as_ref())?.into())
    }

    fn gelu(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::gelu(self.inner.as_ref())?.into())
    }

    fn gelu_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::gelu_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn selu<U>(&self, alpha: U, gamma: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Into<Option<Self::OutputMeta>>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::selu(self.inner.as_ref(), alpha, gamma)?.into())
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
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::selu_(
            self.inner.as_ref(),
            alpha,
            gamma,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::hard_sigmoid(self.inner.as_ref())?.into())
    }

    fn hard_sigmoid_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::hard_sigmoid_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::hard_swish(self.inner.as_ref())?.into())
    }

    fn hard_swish_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::hard_swish_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn softplus(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::softplus(self.inner.as_ref())?.into())
    }

    fn softplus_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::softplus_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn softsign(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::softsign(self.inner.as_ref())?.into())
    }

    fn softsign_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::softsign_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn mish(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::mish(self.inner.as_ref())?.into())
    }

    fn mish_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::mish_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::cbrt(self.inner.as_ref())?.into())
    }

    fn cbrt_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::cbrt_(
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
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::exp10(self.inner.as_ref())?.into())
    }

    fn exp10_<U>(&self, mut out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::exp10_(
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
        Ok(_Tensor::<T, Cpu, DEVICE, Al>::erf_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }
}

// impl<T: CommonBounds, const DEVICE: usize> FloatUnaryOps for DiffTensor<T, Cpu, DEVICE>
// where
//     T: FloatOutUnary + PartialOrd + num_traits::Zero + Cmp<Output = bool> + Cast<FloatUnaryType<T>>,
//     FloatUnaryType<T>:
//         CommonBounds + FloatOutUnary<Output = <T as FloatOutUnary>::Output> + Cast<T>,
//     f64: Cast<<T as FloatOutUnary>::Output>,
//     T::Vec: FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec>,
// {
//     type Output = DiffTensor<FloatUnaryType<T>, Cpu, DEVICE>;

//     type InplaceOutput = Tensor<FloatUnaryType<T>, Cpu, DEVICE>;

//     type OutputMeta = FloatUnaryType<T>;

//     fn sin(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.sin()?;
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             *res = g._mul(x._cos()).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn sin_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.sin_(out)
//     }

//     fn cos(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn tan(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn asin(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn acos(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn atan(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn sinh(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn cosh(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn tanh(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn asinh(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn acosh(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn atanh(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn cos_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.cos_(out)
//     }

//     fn tan_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.tan_(out)
//     }

//     fn asin_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.asin_(out)
//     }

//     fn acos_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.acos_(out)
//     }

//     fn atan_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.atan_(out)
//     }

//     fn sinh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.sinh_(out)
//     }

//     fn cosh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.cosh_(out)
//     }

//     fn tanh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.tanh_(out)
//     }

//     fn asinh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.asinh_(out)
//     }

//     fn acosh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.acosh_(out)
//     }

//     fn atanh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.atanh_(out)
//     }

//     fn exp(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn exp_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.exp_(out)
//     }

//     fn exp2(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.exp2()?;
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             let a: <T as FloatOutUnary>::Output = x._exp2();
//                             let mul: <T as FloatOutUnary>::Output = a._mul(2.0.cast()._ln());
//                             *res = g._mul(mul).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn exp2_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.exp2_(out)
//     }

//     fn sqrt(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.sqrt()?;
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             let a: <T as FloatOutUnary>::Output = x._sqrt();
//                             let mul: <T as FloatOutUnary>::Output = a._mul(2.0.cast());
//                             let c: <T as FloatOutUnary>::Output = mul._recip();
//                             *res = g._mul(c).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn sqrt_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.sqrt_(out)
//     }

//     fn recip(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.recip()?;
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             let mul = x._neg();
//                             let b = x._mul(mul);
//                             let c: <T as FloatOutUnary>::Output = b._recip();
//                             *res = g._mul(c).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn recip_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.recip_(out)
//     }

//     fn ln(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.cbrt()?;
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             let c: <T as FloatOutUnary>::Output = x._recip();
//                             *res = g._mul(c).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn ln_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.ln_(out)
//     }

//     fn log2(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.log2()?; // Ensure this computes the forward pass of log10
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Compute the natural logarithm of 10
//                             let ln10: <T as FloatOutUnary>::Output = 2.0.cast()._ln(); // ln(10)

//                             // Compute the denominator: x * ln(10)
//                             let denom: <T as FloatOutUnary>::Output = x._mul(ln10);

//                             // Compute the gradient of log10: f'(x) = 1 / (x * ln(10))
//                             let log2_grad: <T as FloatOutUnary>::Output = denom._recip();

//                             // Multiply the gradient by the incoming gradient
//                             *res = g._mul(log2_grad).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn log2_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.log2_(out)
//     }

//     fn log10(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.log10()?; // Ensure this computes the forward pass of log10
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Compute the natural logarithm of 10
//                             let ln10: <T as FloatOutUnary>::Output = 10.0.cast()._ln(); // ln(10)

//                             // Compute the denominator: x * ln(10)
//                             let denom: <T as FloatOutUnary>::Output = x._mul(ln10);

//                             // Compute the gradient of log10: f'(x) = 1 / (x * ln(10))
//                             let log10_grad: <T as FloatOutUnary>::Output = denom._recip();

//                             // Multiply the gradient by the incoming gradient
//                             *res = g._mul(log10_grad).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn log10_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.log10_(out)
//     }

//     fn celu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.celu(alpha)?; // Ensure this computes the forward pass of CELU
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Compute the gradient of CELU
//                             let celu_grad: <T as FloatOutUnary>::Output = if x._gt(T::ZERO) {
//                                 T::ONE.cast() // Gradient is 1 if x > 0
//                             } else {
//                                 (x._div(alpha))._exp() // Gradient is e^(x / alpha) if x <= 0
//                             };

//                             // Multiply by the incoming gradient
//                             *res = g._mul(celu_grad).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn celu_<U>(
//         &self,
//         alpha: Self::OutputMeta,
//         out: U,
//     ) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.celu_(alpha, out)
//     }

//     fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.sigmoid()?; // Ensure this computes the forward pass of Sigmoid
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Compute the sigmoid value: f(x) = 1 / (1 + e^(-x))
//                             let sigmoid_value: T = *res; // Use the forward pass result
//                             let sigmoid_value: FloatUnaryType<T> = sigmoid_value.cast();
//                             // Compute the gradient of sigmoid: f'(x) = f(x) * (1 - f(x))
//                             let sigmoid_grad: FloatUnaryType<T> =
//                                 sigmoid_value._mul(FloatUnaryType::<T>::ONE._sub(sigmoid_value)); // f'(x) = f(x) * (1 - f(x))

//                             // Multiply by the incoming gradient
//                             let grad: T = g._mul(sigmoid_grad).cast();
//                             *res = grad;
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn sigmoid_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.sigmoid_(out)
//     }

//     fn elu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.elu(alpha)?; // Ensure this computes the forward pass of ELU
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Extract alpha as a scalar
//                             let alpha: <T as FloatOutUnary>::Output = alpha.into().unwrap();

//                             // Compute the gradient of ELU
//                             let elu_grad: <T as FloatOutUnary>::Output = if x.cast() > 0.0 {
//                                 1.0.cast() // Gradient is 1 if x > 0
//                             } else {
//                                 alpha * x._exp() // Gradient is alpha * exp(x) if x <= 0
//                             };

//                             // Multiply by the incoming gradient
//                             *res = g._mul(elu_grad).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn elu_<U>(
//         &self,
//         alpha: Self::OutputMeta,
//         out: U,
//     ) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.elu_(alpha, out)
//     }

//     fn erf(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.erf()?; // Ensure this computes the forward pass of erf
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Compute the gradient of erf: f'(x) = (2 / sqrt(pi)) * exp(-x^2)
//                             let factor: <T as FloatOutUnary>::Output =
//                                 (2.0 / std::f64::consts::PI.sqrt()).cast(); // 2 / sqrt(pi)
//                             let exp_term: <T as FloatOutUnary>::Output = x._square()._neg()._exp(); // exp(-x^2)
//                             let erf_grad: <T as FloatOutUnary>::Output = factor._mul(exp_term); // (2 / sqrt(pi)) * exp(-x^2)

//                             // Multiply by the incoming gradient
//                             *res = g._mul(erf_grad).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn gelu(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.mish()?;
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             let a: <T as FloatOutUnary>::Output = x._gelu();
//                             *res = g._mul(a).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn gelu_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.gelu_(out)
//     }

//     fn selu<U>(&self, alpha: U, gamma: U) -> std::result::Result<Self::Output, TensorError>
//     where
//         U: Into<Option<Self::OutputMeta>>,
//     {
//         let res = self.inner.selu(gamma, alpha)?;
//         *self.out_degree.as_ref().borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Extract gamma and alpha as scalars
//                             let gamma: <T as FloatOutUnary>::Output = gamma.into().unwrap();
//                             let alpha: <T as FloatOutUnary>::Output = alpha.into().unwrap();

//                             // Compute the SELU gradient
//                             let f_prime_x: <T as FloatOutUnary>::Output = if x.cast() > 0.0 {
//                                 gamma // For x > 0, gradient is gamma
//                             } else {
//                                 gamma * alpha * x._exp() // For x <= 0, gradient is gamma * alpha * exp(x)
//                             };

//                             // Multiply the gradient by the incoming gradient
//                             *res = g._mul(f_prime_x).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn selu_<U>(
//         &self,
//         alpha: Option<Self::OutputMeta>,
//         gamma: Option<Self::OutputMeta>,
//         out: U,
//     ) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.selu_(gamma, alpha, out)
//     }

//     fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.hard_sigmoid()?; // Ensure this computes the Hard Sigmoid forward pass
//         *self.out_degree.borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Define alpha and beta
//                             let alpha: <T as FloatOutUnary>::Output = 0.2.cast(); // Example value for alpha
//                             let beta: <T as FloatOutUnary>::Output = 0.5.cast(); // Example value for beta

//                             // Compute the linear transformation: alpha * x + beta
//                             let lin: <T as FloatOutUnary>::Output = x._mul(alpha)._add(beta);

//                             // Apply the gradient logic:
//                             // If 0 < alpha * x + beta < 1, gradient is alpha; otherwise, it's 0
//                             let f_prime_x: <T as FloatOutUnary>::Output =
//                                 if lin > 0.0.cast() && lin < 1.0.cast() {
//                                     alpha // Gradient is alpha within the range
//                                 } else {
//                                     0.0.cast() // Gradient is 0 outside the range
//                                 };

//                             // Multiply the gradient by the incoming gradient
//                             *res = g._mul(f_prime_x).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn hard_sigmoid_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.hard_sigmoid_(out)
//     }

//     fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.softplus()?; // Ensure this computes the forward pass of Softplus
//         *self.out_degree.borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Compute the Softplus gradient: f'(x) = e^x / (1 + e^x)
//                             let exp_x: <T as FloatOutUnary>::Output = x._exp(); // e^x
//                             let denom: <T as FloatOutUnary>::Output = exp_x._add(1.0.cast()); // 1 + e^x
//                             let softplus_grad: <T as FloatOutUnary>::Output = exp_x._div(denom); // e^x / (1 + e^x)

//                             // Multiply by the incoming gradient
//                             *res = g._mul(softplus_grad).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn hard_swish_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.hard_swish_(out)
//     }

//     fn softplus(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.softplus()?;
//         *self.out_degree.borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             let a: <T as FloatOutUnary>::Output = x._exp();
//                             let add: <T as FloatOutUnary>::Output = a._add(1.0.cast());
//                             *res = g._mul(add).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn softplus_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.softplus_(out)
//     }

//     fn softsign(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.softsign()?;
//         *self.out_degree.borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Compute |x| (absolute value of x)
//                             let abs_x: <T as FloatOutUnary>::Output = x._abs().cast();

//                             // Compute (1 + |x|)^2
//                             let denom: <T as FloatOutUnary>::Output =
//                                 abs_x._add(1.0.cast())._square();

//                             // Compute the gradient: f'(x) = 1 / (1 + |x|)^2
//                             let f_prime_x: <T as FloatOutUnary>::Output = denom._recip();

//                             // Multiply the gradient by the incoming gradient (g)
//                             *res = g._mul(f_prime_x).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn softsign_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.softsign_(out)
//     }

//     fn mish(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.mish()?;
//         *self.out_degree.borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             // Compute u = ln(1 + e^x)
//                             let u: <T as FloatOutUnary>::Output = x._softplus();

//                             // Compute tanh(u)
//                             let tanh_u: <T as FloatOutUnary>::Output = u._tanh();

//                             // Compute 1 - tanh^2(u)
//                             let one_minus_tanh_sq: <T as FloatOutUnary>::Output =
//                                 tanh_u._square()._neg_add(1.0.into());

//                             // Compute du/dx = e^x / (1 + e^x)
//                             let du_dx: <T as FloatOutUnary>::Output =
//                                 x._exp()._div(x._exp()._add(1.0.into()));

//                             // Combine terms: g'(x) = (1 - tanh^2(u)) * du/dx
//                             let g_prime_x: <T as FloatOutUnary>::Output =
//                                 one_minus_tanh_sq._mul(du_dx);

//                             // Combine with product rule: f'(x) = x * g'(x) + tanh(u)
//                             let f_prime_x: <T as FloatOutUnary>::Output =
//                                 x._mul(g_prime_x)._add(tanh_u);

//                             *res = g._mul(f_prime_x).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn mish_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.mish_(out)
//     }

//     fn cbrt(&self) -> std::result::Result<Self::Output, TensorError> {
//         let res = self.inner.cbrt()?;
//         *self.out_degree.borrow_mut() += 1;
//         let mut operand = self.clone();
//         Ok(DiffTensor {
//             inner: res,
//             grad: Rc::new(RefCell::new(None)),
//             out_degree: Rc::new(RefCell::new(0)),
//             backward: Rc::new(RefCell::new(
//                 move |grad: Tensor<FloatUnaryType<T>, Cpu, DEVICE>| {
//                     let new_grad = grad
//                         .inner
//                         .par_iter()
//                         .zip(operand.inner.inner.par_iter())
//                         .strided_map(|(res, (g, x))| {
//                             let a: <T as FloatOutUnary>::Output = x._cbrt();
//                             let mul: <T as FloatOutUnary>::Output = a._mul(3.0.cast());
//                             let b: <T as FloatOutUnary>::Output = a._mul(mul);
//                             let c: <T as FloatOutUnary>::Output = b._recip();
//                             *res = g._mul(c).cast();
//                         })
//                         .collect::<_Tensor<T, Cpu, DEVICE>>();
//                     handle_grad(&mut operand, new_grad.into(), &[])?;
//                     Ok(false)
//                 },
//             )),
//         })
//     }

//     fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.cbrt_(out)
//     }

//     fn sincos(&self) -> std::result::Result<(Self::Output, Self::Output), TensorError> {
//         todo!()
//     }

//     fn sincos_<U, O>(
//         &self,
//         outs: (U, O),
//     ) -> std::result::Result<(Self::Output, Self::Output), TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//         O: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.sincos_(outs)
//     }

//     fn exp10(&self) -> std::result::Result<Self::Output, TensorError> {
//         todo!()
//     }

//     fn exp10_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.exp10_(out)
//     }

//     fn erf_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
//     where
//         U: BorrowMut<Self::InplaceOutput>,
//     {
//         self.inner.erf_(out)
//     }
// }
