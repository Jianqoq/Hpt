use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    rc::Rc,
};

use hpt_common::error::base::TensorError;
use hpt_traits::{
    ops::{binary::Matmul, shape_manipulate::ShapeManipulate},
    tensor::CommonBounds,
};
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};

use crate::{
    backends::cpu::{
        tensor_internal::matmul::matmul_with_out, utils::diff::diff_utils::handle_grad,
    },
    tensor::{DiffTensor, Tensor},
};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};

impl<A, B, const DEVICE: usize, Al> Matmul<Tensor<B, Cpu, DEVICE, Al>>
    for Tensor<A, Cpu, DEVICE, Al>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, Al>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, Al>;

    fn matmul(
        &self,
        rhs: Tensor<B, Cpu, DEVICE, Al>,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(matmul_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            None::<Self::Output>,
        )?
        .into())
    }
    fn matmul_<U>(
        &self,
        rhs: Tensor<B, Cpu, DEVICE, Al>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(matmul_with_out(self.inner.as_ref(), rhs.inner.as_ref(), Some(out))?.into())
    }
}

impl<A, B, const DEVICE: usize, Al> Matmul<&Tensor<B, Cpu, DEVICE, Al>>
    for Tensor<A, Cpu, DEVICE, Al>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, Al>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, Al>;

    fn matmul(
        &self,
        rhs: &Tensor<B, Cpu, DEVICE, Al>,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(matmul_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            None::<Self::Output>,
        )?
        .into())
    }

    fn matmul_<U>(
        &self,
        rhs: &Tensor<B, Cpu, DEVICE, Al>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(matmul_with_out(self.inner.as_ref(), rhs.inner.as_ref(), Some(out))?.into())
    }
}

impl<A, B, const DEVICE: usize, Al> Matmul<DiffTensor<B, Cpu, DEVICE, Al>>
    for DiffTensor<A, Cpu, DEVICE, Al>
where
    A: CommonBounds
        + NormalOut<B>
        + Cast<<A as NormalOut<B>>::Output>
        + NormalOut<<A as NormalOut<B>>::Output>
        + Cast<<A as NormalOut<<A as NormalOut<B>>::Output>>::Output>,
    B: CommonBounds
        + Cast<<A as NormalOut<B>>::Output>
        + Cast<<<A as NormalOut<B>>::Output as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds
        + NormalOut<A>
        + NormalOut<B>
        + Cast<<<A as NormalOut<B>>::Output as NormalOut<B>>::Output>
        + Cast<<A as NormalOut<<A as NormalOut<B>>::Output>>::Output>,
    <<A as NormalOut<B>>::Output as NormalOut<B>>::Output:
        CommonBounds + Cast<<A as NormalOut<<A as NormalOut<B>>::Output>>::Output> + Cast<A>,
    <A as NormalOut<<A as NormalOut<B>>::Output>>::Output: CommonBounds + Cast<B>,
    Al: Allocator + 'static + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = DiffTensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, Al>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, Al>;

    fn matmul(
        &self,
        rhs: DiffTensor<B, Cpu, DEVICE, Al>,
    ) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.matmul(&rhs.inner)?;
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, Al>| {
                    let grad_a = grad.matmul(rhs.inner.t()?)?.try_astype::<A>()?;
                    let grad_b = lhs.inner.t()?.matmul(grad)?.try_astype::<B>()?;
                    handle_grad(&mut lhs, grad_a, &[])?;
                    handle_grad(&mut rhs, grad_b, &[])?;
                    Ok(false)
                },
            )),
        })
    }
    fn matmul_<U>(
        &self,
        rhs: DiffTensor<B, Cpu, DEVICE, Al>,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        self.inner.matmul_(&rhs.inner, out)
    }
}
