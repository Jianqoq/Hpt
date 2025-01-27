use crate::ops::cpu::tensor_internal::normal_out_unary::NormalType;
use crate::ops::cpu::utils::binary::binary_normal::*;
use crate::ops::cpu::utils::diff::diff_utils::handle_grad;
use crate::ops::cpu::utils::unary::unary::unary_fn_with_out_simd;
use crate::tensor::DiffTensor;
use crate::tensor_base::_Tensor;
use crate::Cpu;
use crate::Tensor;
use num::complex::{Complex32, Complex64};
use rayon::iter::ParallelIterator;
use std::cell::RefCell;
use std::ops::{Neg, Not};
use std::rc::Rc;
use std::sync::Arc;
use tensor_common::shape::shape_utils::get_broadcast_axes_from;
use tensor_iterator::iterator_traits::ParStridedIteratorZip;
use tensor_iterator::TensorIterator;
use tensor_traits::tensor::{CommonBounds, TensorInfo};
use tensor_traits::NormalUaryOps;
use tensor_traits::TensorLike;
use tensor_types::convertion::Convertor;
use tensor_types::dtype::TypeCommon;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::{
    BitWiseOut, FloatOutBinary, FloatOutUnary, NormalOut, NormalOutUnary,
};

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Add];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Add];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Add];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Add];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn add(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._add(y),
            |x, y| x._add(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Add];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Add];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Add];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Add];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn add(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().add(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Add];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Add];
    [Tensor]    [i8]        [Tensor]    [std::ops::Add];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Add];
    [Tensor]    [i16]       [Tensor]    [std::ops::Add];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Add];
    [Tensor]    [i32]       [Tensor]    [std::ops::Add];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Add];
    [Tensor]    [i64]       [Tensor]    [std::ops::Add];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Add];
    [Tensor]    [u8]        [Tensor]    [std::ops::Add];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Add];
    [Tensor]    [u16]       [Tensor]    [std::ops::Add];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Add];
    [Tensor]    [u32]       [Tensor]    [std::ops::Add];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Add];
    [Tensor]    [u64]       [Tensor]    [std::ops::Add];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Add];
    [Tensor]    [f32]       [Tensor]    [std::ops::Add];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Add];
    [Tensor]    [f64]       [Tensor]    [std::ops::Add];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Add];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Add];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Add];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Add];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Add];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Add];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Add];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Add];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Add];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<rhs_type>,
    <T as NormalOut<rhs_type>>::Output: CommonBounds,
    <T as NormalOut<rhs_type>>::Output: IntoScalar<<T as NormalOut<rhs_type>>::Output>,
    T::Vec: NormalOut<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as NormalOut<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn add(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().add(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Add];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Add];
    [Tensor]    [i8]        [Tensor]    [std::ops::Add];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Add];
    [Tensor]    [i16]       [Tensor]    [std::ops::Add];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Add];
    [Tensor]    [i32]       [Tensor]    [std::ops::Add];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Add];
    [Tensor]    [i64]       [Tensor]    [std::ops::Add];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Add];
    [Tensor]    [u8]        [Tensor]    [std::ops::Add];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Add];
    [Tensor]    [u16]       [Tensor]    [std::ops::Add];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Add];
    [Tensor]    [u32]       [Tensor]    [std::ops::Add];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Add];
    [Tensor]    [u64]       [Tensor]    [std::ops::Add];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Add];
    [Tensor]    [f32]       [Tensor]    [std::ops::Add];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Add];
    [Tensor]    [f64]       [Tensor]    [std::ops::Add];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Add];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Add];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Add];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Add];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Add];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Add];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Add];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Add];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Add];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: NormalOut<T>,
    <lhs_type as NormalOut<T>>::Output: CommonBounds,
    <lhs_type as NormalOut<T>>::Output: IntoScalar<<T as NormalOut<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: NormalOut<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as NormalOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as NormalOut<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn add(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.add(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type        rhs_type       out_type        trait_name;
    [DiffTensor]    [DiffTensor]   [DiffTensor]    [std::ops::Add];
    [&DiffTensor]   [DiffTensor]   [DiffTensor]    [std::ops::Add];
    [&DiffTensor]   [&DiffTensor]  [DiffTensor]    [std::ops::Add];
    [DiffTensor]    [&DiffTensor]  [DiffTensor]    [std::ops::Add];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
    <T as NormalOut<U>>::Output: IntoScalar<T> + IntoScalar<U>,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn add(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        *self.out_degree.borrow_mut() += 1;
        *rhs.out_degree.borrow_mut() += 1;
        let res = self.inner.clone().add(rhs.inner.clone());
        let lhs_broadcast_axes =
            get_broadcast_axes_from(self.inner.shape(), res.shape()).expect("broadcast failed");
        let rhs_broadcast_axes =
            get_broadcast_axes_from(rhs.inner.shape(), res.shape()).expect("broadcast failed");
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<T as NormalOut<U>>::Output, Cpu, DEVICE>| {
                    let lhs_grad = grad.try_astype::<T>()?;
                    let rhs_grad = grad.try_astype::<U>()?;
                    handle_grad(&mut lhs, lhs_grad, &lhs_broadcast_axes)?;
                    handle_grad(&mut rhs, rhs_grad, &rhs_broadcast_axes)?;
                    Ok(false)
                },
            )),
        }
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Sub];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Sub];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Sub];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Sub];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sub(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._sub(y),
            |x, y| x._sub(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Sub];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Sub];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Sub];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Sub];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sub(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().sub(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Sub];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Sub];
    [Tensor]    [i8]        [Tensor]    [std::ops::Sub];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Sub];
    [Tensor]    [i16]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Sub];
    [Tensor]    [i32]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Sub];
    [Tensor]    [i64]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Sub];
    [Tensor]    [u8]        [Tensor]    [std::ops::Sub];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Sub];
    [Tensor]    [u16]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Sub];
    [Tensor]    [u32]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Sub];
    [Tensor]    [u64]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Sub];
    [Tensor]    [f32]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Sub];
    [Tensor]    [f64]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Sub];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Sub];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Sub];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Sub];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Sub];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Sub];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Sub];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Sub];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Sub];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<rhs_type>,
    <T as NormalOut<rhs_type>>::Output: CommonBounds,
    <T as NormalOut<rhs_type>>::Output: IntoScalar<<T as NormalOut<rhs_type>>::Output>,
    T::Vec: NormalOut<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as NormalOut<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sub(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().sub(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Sub];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Sub];
    [Tensor]    [i8]        [Tensor]    [std::ops::Sub];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Sub];
    [Tensor]    [i16]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Sub];
    [Tensor]    [i32]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Sub];
    [Tensor]    [i64]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Sub];
    [Tensor]    [u8]        [Tensor]    [std::ops::Sub];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Sub];
    [Tensor]    [u16]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Sub];
    [Tensor]    [u32]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Sub];
    [Tensor]    [u64]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Sub];
    [Tensor]    [f32]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Sub];
    [Tensor]    [f64]       [Tensor]    [std::ops::Sub];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Sub];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Sub];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Sub];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Sub];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Sub];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Sub];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Sub];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Sub];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Sub];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: NormalOut<T>,
    <lhs_type as NormalOut<T>>::Output: CommonBounds,
    <lhs_type as NormalOut<T>>::Output: IntoScalar<<T as NormalOut<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: NormalOut<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as NormalOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as NormalOut<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sub(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.sub(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type        rhs_type       out_type        trait_name;
    [DiffTensor]    [DiffTensor]   [DiffTensor]    [std::ops::Sub];
    [&DiffTensor]   [DiffTensor]   [DiffTensor]    [std::ops::Sub];
    [&DiffTensor]   [&DiffTensor]  [DiffTensor]    [std::ops::Sub];
    [DiffTensor]    [&DiffTensor]  [DiffTensor]    [std::ops::Sub];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
    <T as NormalOut<U>>::Output: IntoScalar<T> + IntoScalar<U>,
    Tensor<T, Cpu, DEVICE>: Neg<Output = Tensor<T, Cpu, DEVICE>>,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sub(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        let res = self.inner.clone().sub(rhs.inner.clone());
        let lhs_broadcast_axes =
            get_broadcast_axes_from(self.inner.shape(), res.shape()).expect("broadcast failed");
        let rhs_broadcast_axes =
            get_broadcast_axes_from(rhs.inner.shape(), res.shape()).expect("broadcast failed");
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<T as NormalOut<U>>::Output, Cpu, DEVICE>| {
                    let lhs_grad = -grad.try_astype::<T>()?;
                    let rhs_grad = grad.try_astype::<U>()?;
                    handle_grad(&mut lhs, lhs_grad, &lhs_broadcast_axes)?;
                    handle_grad(&mut rhs, rhs_grad, &rhs_broadcast_axes)?;
                    Ok(false)
                },
            )),
        }
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Div];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Div];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Div];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Div];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + FloatOutBinary<U>,
    U: CommonBounds,
    <T as FloatOutBinary<U>>::Output: CommonBounds,
    <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
    T::Vec: FloatOutBinary<
        <U as TypeCommon>::Vec,
        Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as FloatOutBinary<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn div(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._div(y),
            |x, y| x._div(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Div];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Div];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Div];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Div];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + FloatOutBinary<U>,
    U: CommonBounds,
    <T as FloatOutBinary<U>>::Output: CommonBounds,
    <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
    T::Vec: FloatOutBinary<
        <U as TypeCommon>::Vec,
        Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as FloatOutBinary<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn div(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().div(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Div];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Div];
    [Tensor]    [i8]        [Tensor]    [std::ops::Div];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Div];
    [Tensor]    [i16]       [Tensor]    [std::ops::Div];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Div];
    [Tensor]    [i32]       [Tensor]    [std::ops::Div];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Div];
    [Tensor]    [i64]       [Tensor]    [std::ops::Div];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Div];
    [Tensor]    [u8]        [Tensor]    [std::ops::Div];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Div];
    [Tensor]    [u16]       [Tensor]    [std::ops::Div];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Div];
    [Tensor]    [u32]       [Tensor]    [std::ops::Div];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Div];
    [Tensor]    [u64]       [Tensor]    [std::ops::Div];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Div];
    [Tensor]    [f32]       [Tensor]    [std::ops::Div];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Div];
    [Tensor]    [f64]       [Tensor]    [std::ops::Div];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Div];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Div];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Div];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Div];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Div];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Div];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Div];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Div];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Div];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + FloatOutBinary<rhs_type>,
    <T as FloatOutBinary<rhs_type>>::Output: CommonBounds,
    <T as FloatOutBinary<rhs_type>>::Output: IntoScalar<<T as FloatOutBinary<rhs_type>>::Output>,
    T::Vec: FloatOutBinary<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as FloatOutBinary<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as FloatOutBinary<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn div(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().div(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Div];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Div];
    [Tensor]    [i8]        [Tensor]    [std::ops::Div];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Div];
    [Tensor]    [i16]       [Tensor]    [std::ops::Div];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Div];
    [Tensor]    [i32]       [Tensor]    [std::ops::Div];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Div];
    [Tensor]    [i64]       [Tensor]    [std::ops::Div];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Div];
    [Tensor]    [u8]        [Tensor]    [std::ops::Div];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Div];
    [Tensor]    [u16]       [Tensor]    [std::ops::Div];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Div];
    [Tensor]    [u32]       [Tensor]    [std::ops::Div];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Div];
    [Tensor]    [u64]       [Tensor]    [std::ops::Div];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Div];
    [Tensor]    [f32]       [Tensor]    [std::ops::Div];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Div];
    [Tensor]    [f64]       [Tensor]    [std::ops::Div];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Div];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Div];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Div];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Div];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Div];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Div];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Div];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Div];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Div];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: FloatOutBinary<T>,
    <lhs_type as FloatOutBinary<T>>::Output: CommonBounds,
    <lhs_type as FloatOutBinary<T>>::Output: IntoScalar<<T as FloatOutBinary<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: FloatOutBinary<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as FloatOutBinary<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as FloatOutBinary<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn div(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.div(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type        rhs_type       out_type        trait_name;
    [DiffTensor]    [DiffTensor]   [DiffTensor]    [std::ops::Div];
    [&DiffTensor]   [DiffTensor]   [DiffTensor]    [std::ops::Div];
    [&DiffTensor]   [&DiffTensor]  [DiffTensor]    [std::ops::Div];
    [DiffTensor]    [&DiffTensor]  [DiffTensor]    [std::ops::Div];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>>
    for lhs_type<T, Cpu, DEVICE>
    where
        T: CommonBounds + FloatOutBinary<U>,
        U: CommonBounds,
        <T as FloatOutBinary<U>>::Output: CommonBounds + IntoScalar<U>,
        <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output> +
            FloatOutBinary<U>,
        T::Vec: FloatOutBinary<
            <U as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec
        >,
        <<T as FloatOutBinary<U>>::Output as FloatOutBinary<U>>::Output: CommonBounds +
            IntoScalar<T>,
        <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec: FloatOutBinary<
            <U as TypeCommon>::Vec,
            Output = <<<T as FloatOutBinary<U>>::Output as FloatOutBinary<U>>::Output as TypeCommon>::Vec
        >
{
    type Output = out_type<<T as FloatOutBinary<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn div(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        let res = self.inner.clone().div(rhs.inner.clone());
        let lhs_broadcast_axes = get_broadcast_axes_from(self.inner.shape(), res.shape()).expect(
            "broadcast failed"
        );
        let rhs_broadcast_axes = get_broadcast_axes_from(rhs.inner.shape(), res.shape()).expect(
            "broadcast failed"
        );
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(
                RefCell::new(move |grad: Tensor<<T as FloatOutBinary<U>>::Output, Cpu, DEVICE>| {
                    let lhs_grad = binary_fn_with_out_simd(
                        &grad.inner.as_ref(),
                        rhs.inner.inner.as_ref(),
                        |lhs, rhs| lhs._div(rhs),
                        |lhs, rhs| lhs._div(rhs),
                        None::<
                            Tensor<
                                <<T as FloatOutBinary<U>>::Output as FloatOutBinary<U>>::Output,
                                Cpu,
                                DEVICE
                            >
                        >
                    )?.try_astype::<T>()?;
                    let rhs_grad = binary_fn_with_out_simd_3(
                        &grad.inner.as_ref(),
                        &lhs.inner.inner.as_ref(),
                        rhs.inner.inner.as_ref(),
                        |grad, b, c| {
                            let tmp: <T as FloatOutBinary<U>>::Output = b._neg()._div(c._square());
                            grad._mul(tmp)
                        },
                        |grad, b, c| { grad._mul(b._neg()._div(c._square())) },
                        None::<Tensor<<T as FloatOutBinary<U>>::Output, Cpu, DEVICE>>
                    )?.try_astype::<U>()?;
                    handle_grad(&mut lhs, lhs_grad.into(), &lhs_broadcast_axes)?;
                    handle_grad(&mut rhs, rhs_grad.into(), &rhs_broadcast_axes)?;
                    Ok(false)
                })
            ),
        }
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Mul];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Mul];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Mul];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Mul];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mul(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._mul(y),
            |x, y| x._mul(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Mul];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Mul];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Mul];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Mul];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mul(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().mul(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Mul];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Mul];
    [Tensor]    [i8]        [Tensor]    [std::ops::Mul];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Mul];
    [Tensor]    [i16]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Mul];
    [Tensor]    [i32]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Mul];
    [Tensor]    [i64]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Mul];
    [Tensor]    [u8]        [Tensor]    [std::ops::Mul];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Mul];
    [Tensor]    [u16]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Mul];
    [Tensor]    [u32]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Mul];
    [Tensor]    [u64]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Mul];
    [Tensor]    [f32]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Mul];
    [Tensor]    [f64]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Mul];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Mul];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Mul];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Mul];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Mul];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Mul];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Mul];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Mul];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Mul];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<rhs_type>,
    <T as NormalOut<rhs_type>>::Output: CommonBounds,
    <T as NormalOut<rhs_type>>::Output: IntoScalar<<T as NormalOut<rhs_type>>::Output>,
    T::Vec: NormalOut<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as NormalOut<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mul(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().mul(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Mul];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Mul];
    [Tensor]    [i8]        [Tensor]    [std::ops::Mul];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Mul];
    [Tensor]    [i16]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Mul];
    [Tensor]    [i32]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Mul];
    [Tensor]    [i64]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Mul];
    [Tensor]    [u8]        [Tensor]    [std::ops::Mul];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Mul];
    [Tensor]    [u16]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Mul];
    [Tensor]    [u32]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Mul];
    [Tensor]    [u64]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Mul];
    [Tensor]    [f32]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Mul];
    [Tensor]    [f64]       [Tensor]    [std::ops::Mul];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Mul];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Mul];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Mul];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Mul];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Mul];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Mul];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Mul];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Mul];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Mul];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: NormalOut<T>,
    <lhs_type as NormalOut<T>>::Output: CommonBounds,
    <lhs_type as NormalOut<T>>::Output: IntoScalar<<T as NormalOut<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: NormalOut<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as NormalOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as NormalOut<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mul(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.mul(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type        rhs_type       out_type        trait_name;
    [DiffTensor]    [DiffTensor]   [DiffTensor]    [std::ops::Mul];
    [&DiffTensor]   [DiffTensor]   [DiffTensor]    [std::ops::Mul];
    [&DiffTensor]   [&DiffTensor]  [DiffTensor]    [std::ops::Mul];
    [DiffTensor]    [&DiffTensor]  [DiffTensor]    [std::ops::Mul];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
    <T as NormalOut<U>>::Output: IntoScalar<T> + IntoScalar<U> + NormalOut<U> + NormalOut<T>,
    <<T as NormalOut<U>>::Output as NormalOut<U>>::Output: CommonBounds + IntoScalar<T>,
    <<T as NormalOut<U>>::Output as NormalOut<T>>::Output: CommonBounds + IntoScalar<U>,
    <<T as NormalOut<U>>::Output as TypeCommon>::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<<T as NormalOut<U>>::Output as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
    <<T as NormalOut<U>>::Output as TypeCommon>::Vec: NormalOut<
        <T as TypeCommon>::Vec,
        Output = <<<T as NormalOut<U>>::Output as NormalOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mul(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        *self.out_degree.borrow_mut() += 1;
        *rhs.out_degree.borrow_mut() += 1;
        let res = self.inner.clone().mul(rhs.inner.clone());
        let lhs_broadcast_axes =
            get_broadcast_axes_from(self.inner.shape(), res.shape()).expect("broadcast failed");
        let rhs_broadcast_axes =
            get_broadcast_axes_from(rhs.inner.shape(), res.shape()).expect("broadcast failed");
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<T as NormalOut<U>>::Output, Cpu, DEVICE>| {
                    let lhs_grad = (grad.clone() * rhs.inner.clone()).try_astype::<T>()?;
                    let rhs_grad = (grad.clone() * lhs.inner.clone()).try_astype::<U>()?;
                    handle_grad(&mut lhs, lhs_grad, &lhs_broadcast_axes)?;
                    handle_grad(&mut rhs, rhs_grad, &rhs_broadcast_axes)?;
                    Ok(false)
                },
            )),
        }
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Rem];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Rem];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Rem];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Rem];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn rem(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._rem(y),
            |x, y| x._rem(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Rem];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Rem];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Rem];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Rem];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn rem(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().rem(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Rem];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Rem];
    [Tensor]    [i8]        [Tensor]    [std::ops::Rem];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Rem];
    [Tensor]    [i16]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Rem];
    [Tensor]    [i32]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Rem];
    [Tensor]    [i64]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Rem];
    [Tensor]    [u8]        [Tensor]    [std::ops::Rem];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Rem];
    [Tensor]    [u16]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Rem];
    [Tensor]    [u32]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Rem];
    [Tensor]    [u64]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Rem];
    [Tensor]    [f32]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Rem];
    [Tensor]    [f64]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Rem];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Rem];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Rem];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Rem];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Rem];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Rem];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Rem];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Rem];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Rem];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<rhs_type>,
    <T as NormalOut<rhs_type>>::Output: CommonBounds,
    <T as NormalOut<rhs_type>>::Output: IntoScalar<<T as NormalOut<rhs_type>>::Output>,
    T::Vec: NormalOut<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as NormalOut<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as NormalOut<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn rem(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().rem(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Rem];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Rem];
    [Tensor]    [i8]        [Tensor]    [std::ops::Rem];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Rem];
    [Tensor]    [i16]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Rem];
    [Tensor]    [i32]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Rem];
    [Tensor]    [i64]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Rem];
    [Tensor]    [u8]        [Tensor]    [std::ops::Rem];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Rem];
    [Tensor]    [u16]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Rem];
    [Tensor]    [u32]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Rem];
    [Tensor]    [u64]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Rem];
    [Tensor]    [f32]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [f32]       [Tensor]    [std::ops::Rem];
    [Tensor]    [f64]       [Tensor]    [std::ops::Rem];
    [&Tensor]   [f64]       [Tensor]    [std::ops::Rem];
    [Tensor]    [Complex32] [Tensor]    [std::ops::Rem];
    [&Tensor]   [Complex32] [Tensor]    [std::ops::Rem];
    [Tensor]    [Complex64] [Tensor]    [std::ops::Rem];
    [&Tensor]   [Complex64] [Tensor]    [std::ops::Rem];
    [Tensor]    [half::f16] [Tensor]    [std::ops::Rem];
    [&Tensor]   [half::f16] [Tensor]    [std::ops::Rem];
    [Tensor]    [half::bf16][Tensor]    [std::ops::Rem];
    [&Tensor]   [half::bf16][Tensor]    [std::ops::Rem];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: NormalOut<T>,
    <lhs_type as NormalOut<T>>::Output: CommonBounds,
    <lhs_type as NormalOut<T>>::Output: IntoScalar<<T as NormalOut<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: NormalOut<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as NormalOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as NormalOut<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn rem(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.rem(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type        rhs_type       out_type        trait_name;
    [DiffTensor]    [DiffTensor]   [DiffTensor]    [std::ops::Rem];
    [&DiffTensor]   [DiffTensor]   [DiffTensor]    [std::ops::Rem];
    [&DiffTensor]   [&DiffTensor]  [DiffTensor]    [std::ops::Rem];
    [DiffTensor]    [&DiffTensor]  [DiffTensor]    [std::ops::Rem];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds + IntoScalar<T> + FloatOutBinary<U>,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
    T::Vec: NormalOut<
        <U as TypeCommon>::Vec,
        Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec,
    >,
    <<T as NormalOut<U>>::Output as FloatOutBinary<U>>::Output: CommonBounds + NormalOutUnary,
    <T as NormalOut<U>>::Output:
        NormalOut<<<T as NormalOut<U>>::Output as FloatOutBinary<U>>::Output>,
    <<T as NormalOut<U>>::Output as NormalOut<
        <<T as NormalOut<U>>::Output as FloatOutBinary<U>>::Output,
    >>::Output: FloatOutUnary + CommonBounds + IntoScalar<U>,
{
    type Output = out_type<<T as NormalOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn rem(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        *self.out_degree.borrow_mut() += 1;
        *rhs.out_degree.borrow_mut() += 1;
        let res = self.inner.clone().rem(rhs.inner.clone());
        let lhs_broadcast_axes =
            get_broadcast_axes_from(self.inner.shape(), res.shape()).expect("broadcast failed");
        let rhs_broadcast_axes =
            get_broadcast_axes_from(rhs.inner.shape(), res.shape()).expect("broadcast failed");
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<T as NormalOut<U>>::Output, Cpu, DEVICE>| {
                    handle_grad(&mut lhs, grad.try_astype::<T>()?, &lhs_broadcast_axes)?;

                    let rhs_grad: _Tensor<U, Cpu, DEVICE> = grad
                        .inner
                        .par_iter()
                        .zip(rhs.inner.inner.par_iter())
                        .strided_map(|(res, (x, y))| {
                            let div = x._div(y);
                            let floor_div = div._floor();
                            *res = x._neg()._mul(floor_div).into_scalar();
                        })
                        .collect();

                    handle_grad(&mut rhs, rhs_grad.into(), &rhs_broadcast_axes)?;
                    Ok(false)
                },
            )),
        }
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::BitAnd];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::BitAnd];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::BitAnd];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::BitAnd];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitand(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._bitand(y),
            |x, y| x._bitand(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::BitAnd];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::BitAnd];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitand(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().bitand(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [bool]      [Tensor]    [std::ops::BitAnd];
    [Tensor]    [i8]        [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [i8]        [Tensor]    [std::ops::BitAnd];
    [Tensor]    [i16]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [i16]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [i32]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [i32]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [i64]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [i64]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [u8]        [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [u8]        [Tensor]    [std::ops::BitAnd];
    [Tensor]    [u16]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [u16]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [u32]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [u32]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [u64]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [u64]       [Tensor]    [std::ops::BitAnd];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<rhs_type>,
    <T as BitWiseOut<rhs_type>>::Output: CommonBounds,
    <T as BitWiseOut<rhs_type>>::Output: IntoScalar<<T as BitWiseOut<rhs_type>>::Output>,
    T::Vec: BitWiseOut<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitand(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().bitand(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [bool]      [Tensor]    [std::ops::BitAnd];
    [Tensor]    [i8]        [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [i8]        [Tensor]    [std::ops::BitAnd];
    [Tensor]    [i16]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [i16]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [i32]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [i32]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [i64]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [i64]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [u8]        [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [u8]        [Tensor]    [std::ops::BitAnd];
    [Tensor]    [u16]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [u16]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [u32]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [u32]       [Tensor]    [std::ops::BitAnd];
    [Tensor]    [u64]       [Tensor]    [std::ops::BitAnd];
    [&Tensor]   [u64]       [Tensor]    [std::ops::BitAnd];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: BitWiseOut<T>,
    <lhs_type as BitWiseOut<T>>::Output: CommonBounds,
    <lhs_type as BitWiseOut<T>>::Output: IntoScalar<<lhs_type as BitWiseOut<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: BitWiseOut<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as BitWiseOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as BitWiseOut<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitand(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.bitand(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::BitOr];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::BitOr];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::BitOr];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::BitOr];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitor(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._bitor(y),
            |x, y| x._bitor(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::BitOr];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::BitOr];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::BitOr];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::BitOr];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitor(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().bitor(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::BitOr];
    [&Tensor]   [bool]      [Tensor]    [std::ops::BitOr];
    [Tensor]    [i8]        [Tensor]    [std::ops::BitOr];
    [&Tensor]   [i8]        [Tensor]    [std::ops::BitOr];
    [Tensor]    [i16]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [i16]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [i32]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [i32]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [i64]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [i64]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [u8]        [Tensor]    [std::ops::BitOr];
    [&Tensor]   [u8]        [Tensor]    [std::ops::BitOr];
    [Tensor]    [u16]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [u16]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [u32]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [u32]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [u64]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [u64]       [Tensor]    [std::ops::BitOr];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<rhs_type>,
    <T as BitWiseOut<rhs_type>>::Output: CommonBounds,
    <T as BitWiseOut<rhs_type>>::Output: IntoScalar<<T as BitWiseOut<rhs_type>>::Output>,
    T::Vec: BitWiseOut<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitor(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().bitor(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::BitOr];
    [&Tensor]   [bool]      [Tensor]    [std::ops::BitOr];
    [Tensor]    [i8]        [Tensor]    [std::ops::BitOr];
    [&Tensor]   [i8]        [Tensor]    [std::ops::BitOr];
    [Tensor]    [i16]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [i16]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [i32]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [i32]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [i64]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [i64]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [u8]        [Tensor]    [std::ops::BitOr];
    [&Tensor]   [u8]        [Tensor]    [std::ops::BitOr];
    [Tensor]    [u16]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [u16]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [u32]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [u32]       [Tensor]    [std::ops::BitOr];
    [Tensor]    [u64]       [Tensor]    [std::ops::BitOr];
    [&Tensor]   [u64]       [Tensor]    [std::ops::BitOr];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: BitWiseOut<T>,
    <lhs_type as BitWiseOut<T>>::Output: CommonBounds,
    <lhs_type as BitWiseOut<T>>::Output: IntoScalar<<lhs_type as BitWiseOut<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: BitWiseOut<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as BitWiseOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as BitWiseOut<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitor(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.bitor(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::BitXor];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::BitXor];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::BitXor];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::BitXor];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitxor(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._bitxor(y),
            |x, y| x._bitxor(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::BitXor];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::BitXor];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::BitXor];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::BitXor];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitxor(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().bitxor(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::BitXor];
    [&Tensor]   [bool]      [Tensor]    [std::ops::BitXor];
    [Tensor]    [i8]        [Tensor]    [std::ops::BitXor];
    [&Tensor]   [i8]        [Tensor]    [std::ops::BitXor];
    [Tensor]    [i16]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [i16]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [i32]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [i32]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [i64]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [i64]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [u8]        [Tensor]    [std::ops::BitXor];
    [&Tensor]   [u8]        [Tensor]    [std::ops::BitXor];
    [Tensor]    [u16]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [u16]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [u32]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [u32]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [u64]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [u64]       [Tensor]    [std::ops::BitXor];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<rhs_type>,
    <T as BitWiseOut<rhs_type>>::Output: CommonBounds,
    <T as BitWiseOut<rhs_type>>::Output: IntoScalar<<T as BitWiseOut<rhs_type>>::Output>,
    T::Vec: BitWiseOut<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitxor(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().bitxor(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::BitXor];
    [&Tensor]   [bool]      [Tensor]    [std::ops::BitXor];
    [Tensor]    [i8]        [Tensor]    [std::ops::BitXor];
    [&Tensor]   [i8]        [Tensor]    [std::ops::BitXor];
    [Tensor]    [i16]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [i16]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [i32]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [i32]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [i64]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [i64]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [u8]        [Tensor]    [std::ops::BitXor];
    [&Tensor]   [u8]        [Tensor]    [std::ops::BitXor];
    [Tensor]    [u16]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [u16]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [u32]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [u32]       [Tensor]    [std::ops::BitXor];
    [Tensor]    [u64]       [Tensor]    [std::ops::BitXor];
    [&Tensor]   [u64]       [Tensor]    [std::ops::BitXor];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: BitWiseOut<T>,
    <lhs_type as BitWiseOut<T>>::Output: CommonBounds,
    <lhs_type as BitWiseOut<T>>::Output: IntoScalar<<lhs_type as BitWiseOut<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: BitWiseOut<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as BitWiseOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as BitWiseOut<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bitxor(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.bitxor(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Shl];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Shl];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Shl];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Shl];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn shl(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._shl(y),
            |x, y| x._shl(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Shl];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Shl];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Shl];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Shl];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn shl(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().shl(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Shl];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Shl];
    [Tensor]    [i8]        [Tensor]    [std::ops::Shl];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Shl];
    [Tensor]    [i16]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Shl];
    [Tensor]    [i32]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Shl];
    [Tensor]    [i64]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Shl];
    [Tensor]    [u8]        [Tensor]    [std::ops::Shl];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Shl];
    [Tensor]    [u16]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Shl];
    [Tensor]    [u32]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Shl];
    [Tensor]    [u64]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Shl];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<rhs_type>,
    <T as BitWiseOut<rhs_type>>::Output: CommonBounds,
    <T as BitWiseOut<rhs_type>>::Output: IntoScalar<<T as BitWiseOut<rhs_type>>::Output>,
    T::Vec: BitWiseOut<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn shl(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().shl(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Shl];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Shl];
    [Tensor]    [i8]        [Tensor]    [std::ops::Shl];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Shl];
    [Tensor]    [i16]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Shl];
    [Tensor]    [i32]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Shl];
    [Tensor]    [i64]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Shl];
    [Tensor]    [u8]        [Tensor]    [std::ops::Shl];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Shl];
    [Tensor]    [u16]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Shl];
    [Tensor]    [u32]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Shl];
    [Tensor]    [u64]       [Tensor]    [std::ops::Shl];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Shl];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: BitWiseOut<T>,
    <lhs_type as BitWiseOut<T>>::Output: CommonBounds,
    <lhs_type as BitWiseOut<T>>::Output: IntoScalar<<lhs_type as BitWiseOut<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: BitWiseOut<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as BitWiseOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as BitWiseOut<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn shl(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.shl(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::Shr];
    [&_Tensor]   [_Tensor]   [_Tensor]    [std::ops::Shr];
    [&_Tensor]   [&_Tensor]  [_Tensor]    [std::ops::Shr];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::Shr];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn shr(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        return binary_fn_with_out_simd(
            &self,
            &rhs,
            |x, y| x._shr(y),
            |x, y| x._shr(y),
            None::<Self::Output>,
        )
        .unwrap();
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name;
    [Tensor]    [Tensor]   [Tensor]    [std::ops::Shr];
    [&Tensor]   [Tensor]   [Tensor]    [std::ops::Shr];
    [&Tensor]   [&Tensor]  [Tensor]    [std::ops::Shr];
    [Tensor]    [&Tensor]  [Tensor]    [std::ops::Shr];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<U>,
    U: CommonBounds,
    <T as BitWiseOut<U>>::Output: CommonBounds,
    <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
    T::Vec: BitWiseOut<
        <U as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn shr(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        self.inner.as_ref().shr(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Shr];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Shr];
    [Tensor]    [i8]        [Tensor]    [std::ops::Shr];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Shr];
    [Tensor]    [i16]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Shr];
    [Tensor]    [i32]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Shr];
    [Tensor]    [i64]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Shr];
    [Tensor]    [u8]        [Tensor]    [std::ops::Shr];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Shr];
    [Tensor]    [u16]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Shr];
    [Tensor]    [u32]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Shr];
    [Tensor]    [u64]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Shr];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + BitWiseOut<rhs_type>,
    <T as BitWiseOut<rhs_type>>::Output: CommonBounds,
    <T as BitWiseOut<rhs_type>>::Output: IntoScalar<<T as BitWiseOut<rhs_type>>::Output>,
    T::Vec: BitWiseOut<
        <rhs_type as TypeCommon>::Vec,
        Output = <<T as BitWiseOut<rhs_type>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as BitWiseOut<rhs_type>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn shr(self, rhs: rhs_type) -> Self::Output {
        let rhs: _Tensor<rhs_type, Cpu, DEVICE> = rhs.into();
        self.inner.as_ref().shr(rhs).into()
    }
}

#[duplicate::duplicate_item(
    rhs_type    lhs_type    out_type    trait_name;
    [Tensor]    [bool]      [Tensor]    [std::ops::Shr];
    [&Tensor]   [bool]      [Tensor]    [std::ops::Shr];
    [Tensor]    [i8]        [Tensor]    [std::ops::Shr];
    [&Tensor]   [i8]        [Tensor]    [std::ops::Shr];
    [Tensor]    [i16]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [i16]       [Tensor]    [std::ops::Shr];
    [Tensor]    [i32]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [i32]       [Tensor]    [std::ops::Shr];
    [Tensor]    [i64]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [i64]       [Tensor]    [std::ops::Shr];
    [Tensor]    [u8]        [Tensor]    [std::ops::Shr];
    [&Tensor]   [u8]        [Tensor]    [std::ops::Shr];
    [Tensor]    [u16]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [u16]       [Tensor]    [std::ops::Shr];
    [Tensor]    [u32]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [u32]       [Tensor]    [std::ops::Shr];
    [Tensor]    [u64]       [Tensor]    [std::ops::Shr];
    [&Tensor]   [u64]       [Tensor]    [std::ops::Shr];
)]
impl<T, const DEVICE: usize> trait_name<rhs_type<T, Cpu, DEVICE>> for lhs_type
where
    T: CommonBounds,
    lhs_type: BitWiseOut<T>,
    <lhs_type as BitWiseOut<T>>::Output: CommonBounds,
    <lhs_type as BitWiseOut<T>>::Output: IntoScalar<<lhs_type as BitWiseOut<T>>::Output>,
    <lhs_type as TypeCommon>::Vec: BitWiseOut<
        <T as TypeCommon>::Vec,
        Output = <<lhs_type as BitWiseOut<T>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<lhs_type as BitWiseOut<T>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn shr(self, rhs: rhs_type<T, Cpu, DEVICE>) -> Self::Output {
        let lhs: _Tensor<lhs_type, Cpu, DEVICE> = self.into();
        lhs.shr(rhs.inner.as_ref()).into()
    }
}

#[duplicate::duplicate_item(
    lhs_type     rhs_type    out_type     trait_name               trait_method     op;
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::AddAssign]    [add_assign]    [_add];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::AddAssign]    [add_assign]    [_add];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::SubAssign]    [sub_assign]    [_sub];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::SubAssign]    [sub_assign]    [_sub];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::MulAssign]    [mul_assign]    [_mul];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::MulAssign]    [mul_assign]    [_mul];
    [_Tensor]    [_Tensor]   [_Tensor]    [std::ops::RemAssign]    [rem_assign]    [_rem];
    [_Tensor]    [&_Tensor]  [_Tensor]    [std::ops::RemAssign]    [rem_assign]    [_rem];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds + IntoScalar<T>,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn trait_method(&mut self, rhs: rhs_type<U, Cpu, DEVICE>) {
        self.par_iter_mut().zip(rhs.par_iter()).for_each(|(x, y)| {
            *x = x.op(y).into_scalar();
        });
    }
}

#[duplicate::duplicate_item(
    lhs_type    rhs_type        trait_name               trait_method;
    [Tensor]    [Tensor]        [std::ops::AddAssign]    [add_assign];
    [Tensor]    [&Tensor]       [std::ops::AddAssign]    [add_assign];
    [Tensor]    [Tensor]        [std::ops::SubAssign]    [sub_assign];
    [Tensor]    [&Tensor]       [std::ops::SubAssign]    [sub_assign];
    [Tensor]    [Tensor]        [std::ops::MulAssign]    [mul_assign];
    [Tensor]    [&Tensor]       [std::ops::MulAssign]    [mul_assign];
    [Tensor]    [Tensor]        [std::ops::RemAssign]    [rem_assign];
    [Tensor]    [&Tensor]       [std::ops::RemAssign]    [rem_assign];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + NormalOut<U>,
    U: CommonBounds,
    <T as NormalOut<U>>::Output: CommonBounds + IntoScalar<T>,
    <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn trait_method(&mut self, rhs: rhs_type<U, Cpu, DEVICE>) {
        Arc::make_mut(&mut self.inner).trait_method(rhs.inner.as_ref());
    }
}

impl<T, U, const DEVICE: usize> PartialEq<_Tensor<U, Cpu, DEVICE>> for _Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Convertor,
    U: CommonBounds + Convertor,
{
    fn eq(&self, other: &_Tensor<U, Cpu, DEVICE>) -> bool {
        if self.size() != other.size() {
            return false;
        }
        if self.shape() != other.shape() {
            return false;
        }
        self.allclose(other)
    }
}

impl<T> Not for _Tensor<T>
where
    T: BitWiseOut<T> + CommonBounds,
    <T as BitWiseOut>::Output: CommonBounds,
    T::Vec: BitWiseOut<T::Vec, Output = <<T as BitWiseOut>::Output as TypeCommon>::Vec>,
{
    type Output = _Tensor<<T as BitWiseOut<T>>::Output>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn not(self) -> Self::Output {
        let lhs: _Tensor<T> = self.into();
        unary_fn_with_out_simd(
            &lhs,
            |x| x._not(),
            |x| x._not(),
            None::<_Tensor<<T as BitWiseOut<T>>::Output>>,
        )
        .unwrap()
    }
}

impl<T> Not for &_Tensor<T>
where
    T: BitWiseOut<T> + CommonBounds,
    <T as BitWiseOut>::Output: CommonBounds,
    T::Vec: BitWiseOut<T::Vec, Output = <<T as BitWiseOut>::Output as TypeCommon>::Vec>,
{
    type Output = _Tensor<<T as BitWiseOut<T>>::Output>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn not(self) -> Self::Output {
        let lhs: _Tensor<T> = self.into();
        unary_fn_with_out_simd(
            &lhs,
            |x| x._not(),
            |x| x._not(),
            None::<_Tensor<<T as BitWiseOut<T>>::Output>>,
        )
        .unwrap()
    }
}

impl<T> Neg for _Tensor<T>
where
    T: CommonBounds,
    T::Vec: NormalOutUnary,
    T: NormalOutUnary,
    _Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
{
    type Output = _Tensor<NormalType<T>>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn neg(self) -> Self::Output {
        <_Tensor<T> as NormalUaryOps>::neg(&self).unwrap()
    }
}

impl<T> Neg for &_Tensor<T>
where
    T: CommonBounds,
    T::Vec: NormalOutUnary,
    T: NormalOutUnary,
    _Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
{
    type Output = _Tensor<NormalType<T>>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn neg(self) -> Self::Output {
        <_Tensor<T> as NormalUaryOps>::neg(&self).unwrap()
    }
}

impl<T, U> PartialEq<Tensor<U>> for Tensor<T>
where
    T: CommonBounds + Convertor,
    U: CommonBounds + Convertor,
{
    fn eq(&self, other: &Tensor<U>) -> bool {
        if self.size() != other.size() {
            return false;
        }
        if self.shape() != other.shape() {
            return false;
        }
        self.allclose(other)
    }
}

impl<T> Not for Tensor<T>
where
    T: BitWiseOut<T> + CommonBounds,
    <T as BitWiseOut>::Output: CommonBounds,
    T::Vec: BitWiseOut<T::Vec, Output = <<T as BitWiseOut>::Output as TypeCommon>::Vec>,
{
    type Output = Tensor<<T as BitWiseOut<T>>::Output>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn not(self) -> Self::Output {
        self.inner.as_ref().not().into()
    }
}

impl<T> Not for &Tensor<T>
where
    T: BitWiseOut<T> + CommonBounds,
    <T as BitWiseOut>::Output: CommonBounds,
    T::Vec: BitWiseOut<T::Vec, Output = <<T as BitWiseOut>::Output as TypeCommon>::Vec>,
{
    type Output = Tensor<<T as BitWiseOut<T>>::Output>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn not(self) -> Self::Output {
        self.inner.as_ref().not().into()
    }
}

impl<T> Neg for Tensor<T>
where
    T: CommonBounds,
    T::Vec: NormalOutUnary,
    T: NormalOutUnary,
    Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
{
    type Output = Tensor<NormalType<T>>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn neg(self) -> Self::Output {
        <_Tensor<T> as NormalUaryOps>::neg(self.inner.as_ref())
            .unwrap()
            .into()
    }
}

impl<T> Neg for &Tensor<T>
where
    T: CommonBounds,
    T::Vec: NormalOutUnary,
    T: NormalOutUnary,
    Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
{
    type Output = Tensor<NormalType<T>>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn neg(self) -> Self::Output {
        <_Tensor<T> as NormalUaryOps>::neg(self.inner.as_ref())
            .unwrap()
            .into()
    }
}
