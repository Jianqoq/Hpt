use crate::ops::cpu::binary_normal::*;
use crate::tensor::DiffTensor;
use crate::Cpu;
use crate::Tensor;
use std::cell::RefCell;
use std::ops::Neg;
use std::rc::Rc;
use tensor_common::shape::shape_utils::get_broadcast_axes_from;
use tensor_traits::tensor::{CommonBounds, TensorInfo};
use tensor_types::dtype::TypeCommon;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::FloatOutBinary;
use tensor_types::type_promote::NormalOut;
use tensor_types::type_promote::NormalOutUnary;

use super::utils::handle_grad;

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
            grad: None,
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<T as NormalOut<U>>::Output, Cpu, DEVICE>| {
                    let lhs_grad = grad.try_astype::<T>()?;
                    let rhs_grad = grad.try_astype::<U>()?;
                    handle_grad(&mut lhs, lhs_grad, &lhs_broadcast_axes)?;
                    handle_grad(&mut rhs, rhs_grad, &rhs_broadcast_axes)?;
                    Ok(())
                },
            )),
        }
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
            grad: None,
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<T as NormalOut<U>>::Output, Cpu, DEVICE>| {
                    let lhs_grad = -grad.try_astype::<T>()?;
                    let rhs_grad = grad.try_astype::<U>()?;
                    handle_grad(&mut lhs, lhs_grad, &lhs_broadcast_axes)?;
                    handle_grad(&mut rhs, rhs_grad, &rhs_broadcast_axes)?;
                    Ok(())
                },
            )),
        }
    }
}

#[duplicate::duplicate_item(
    lhs_type        rhs_type       out_type        trait_name;
    [DiffTensor]    [DiffTensor]   [DiffTensor]    [std::ops::Div];
    [&DiffTensor]   [DiffTensor]   [DiffTensor]    [std::ops::Div];
    [&DiffTensor]   [&DiffTensor]  [DiffTensor]    [std::ops::Div];
    [DiffTensor]    [&DiffTensor]  [DiffTensor]    [std::ops::Div];
)]
impl<T, U, const DEVICE: usize> trait_name<rhs_type<U, Cpu, DEVICE>> for lhs_type<T, Cpu, DEVICE>
where
    T: CommonBounds + FloatOutBinary<U>,
    U: CommonBounds,
    <T as FloatOutBinary<U>>::Output: CommonBounds + IntoScalar<U>,
    <T as FloatOutBinary<U>>::Output:
        IntoScalar<<T as FloatOutBinary<U>>::Output> + FloatOutBinary<U>,
    T::Vec: FloatOutBinary<
        <U as TypeCommon>::Vec,
        Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec,
    >,
    <<T as FloatOutBinary<U>>::Output as FloatOutBinary<U>>::Output: CommonBounds + IntoScalar<T>,
    <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec: FloatOutBinary<
        <U as TypeCommon>::Vec,
        Output = <<<T as FloatOutBinary<U>>::Output as FloatOutBinary<U>>::Output as TypeCommon>::Vec,
    >,
{
    type Output = out_type<<T as FloatOutBinary<U>>::Output, Cpu, DEVICE>;
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn div(self, rhs: rhs_type<U, Cpu, DEVICE>) -> Self::Output {
        let res = self.inner.clone().div(rhs.inner.clone());
        let lhs_broadcast_axes =
            get_broadcast_axes_from(self.inner.shape(), res.shape()).expect("broadcast failed");
        let rhs_broadcast_axes =
            get_broadcast_axes_from(rhs.inner.shape(), res.shape()).expect("broadcast failed");
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        DiffTensor {
            inner: res,
            grad: None,
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<T as FloatOutBinary<U>>::Output, Cpu, DEVICE>| {
                    let lhs_grad = binary_fn_with_out_simd(
                        &grad.inner.as_ref(),
                        rhs.inner.inner.as_ref(),
                        |lhs, rhs| lhs._div(rhs),
                        |lhs, rhs| lhs._div(rhs),
                        None::<Tensor<<<T as FloatOutBinary<U>>::Output as FloatOutBinary<U>>::Output, Cpu, DEVICE>>,
                    )?.try_astype::<T>()?;
                    let rhs_grad = binary_fn_with_out_simd_3(
                        &grad.inner.as_ref(),
                        &lhs.inner.inner.as_ref(),
                        rhs.inner.inner.as_ref(),
                        |grad, b, c| {
                            let tmp: <T as FloatOutBinary<U>>::Output = b._neg()._div(c._square());
                            grad._mul(tmp)
                        },
                        |grad, b, c| {
                            grad._mul(b._neg()._div(c._square()))
                        },
                        None::<Tensor<<T as FloatOutBinary<U>>::Output, Cpu, DEVICE>>,
                    )?.try_astype::<U>()?;
                    handle_grad(&mut lhs, lhs_grad.into(), &lhs_broadcast_axes)?;
                    handle_grad(&mut rhs, rhs_grad.into(), &rhs_broadcast_axes)?;
                    Ok(())
                },
            )),
        }
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
            grad: None,
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<T as NormalOut<U>>::Output, Cpu, DEVICE>| {
                    let lhs_grad = (grad.clone() * rhs.inner.clone()).try_astype::<T>()?;
                    let rhs_grad = (grad.clone() * lhs.inner.clone()).try_astype::<U>()?;
                    handle_grad(&mut lhs, lhs_grad, &lhs_broadcast_axes)?;
                    handle_grad(&mut rhs, rhs_grad, &rhs_broadcast_axes)?;
                    Ok(())
                },
            )),
        }
    }
}
