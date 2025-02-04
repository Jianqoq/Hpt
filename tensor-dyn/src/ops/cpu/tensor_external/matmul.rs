use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    rc::Rc,
};

use tensor_common::error::base::TensorError;
use tensor_traits::{CommonBounds, Matmul, ShapeManipulate};
use tensor_types::{cast::Cast, type_promote::NormalOut};

use crate::{
    ops::cpu::{tensor_internal::matmul::matmul_with_out, utils::diff::diff_utils::handle_grad},
    tensor::{DiffTensor, Tensor},
};

impl<A, B> Matmul<Tensor<B>> for Tensor<A>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
{
    type Output = Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = Tensor<<A as NormalOut<B>>::Output>;

    fn matmul(&self, rhs: Tensor<B>) -> std::result::Result<Self::Output, TensorError> {
        Ok(matmul_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            None::<Self::Output>,
        )?
        .into())
    }
    fn matmul_<U>(&self, rhs: Tensor<B>, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(matmul_with_out(self.inner.as_ref(), rhs.inner.as_ref(), Some(out))?.into())
    }
}

impl<A, B> Matmul<&Tensor<B>> for Tensor<A>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
{
    type Output = Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = Tensor<<A as NormalOut<B>>::Output>;

    fn matmul(&self, rhs: &Tensor<B>) -> std::result::Result<Self::Output, TensorError> {
        Ok(matmul_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            None::<Self::Output>,
        )?
        .into())
    }

    fn matmul_<U>(&self, rhs: &Tensor<B>, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(matmul_with_out(self.inner.as_ref(), rhs.inner.as_ref(), Some(out))?.into())
    }
}

impl<A, B> Matmul<DiffTensor<B>> for DiffTensor<A>
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
    <<A as NormalOut<B>>::Output as NormalOut<B>>::Output: CommonBounds
        + Cast<<A as NormalOut<<A as NormalOut<B>>::Output>>::Output>
        + Cast<A>,
    <A as NormalOut<<A as NormalOut<B>>::Output>>::Output: CommonBounds + Cast<B>,
{
    type Output = DiffTensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = Tensor<<A as NormalOut<B>>::Output>;

    fn matmul(&self, rhs: DiffTensor<B>) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.matmul(&rhs.inner)?;
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |grad: Tensor<<A as NormalOut<B>>::Output>| {
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
        rhs: DiffTensor<B>,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        self.inner.matmul_(&rhs.inner, out)
    }
}
