use std::borrow::{ Borrow, BorrowMut };

use tensor_traits::{ CommonBounds, Matmul };
use tensor_types::{ into_scalar::IntoScalar, type_promote::NormalOut };

use crate::{ ops::cpu::matmul::matmul_with_out, tensor::Tensor, tensor_base::_Tensor };

impl<A, B> Matmul<Tensor<B>>
    for Tensor<A>
    where
        A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
        B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
        <A as NormalOut<B>>::Output: CommonBounds
{
    type Output = Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    fn matmul(&self, rhs: Tensor<B>) -> anyhow::Result<Self::Output> {
        Ok(matmul_with_out(self, &rhs, None::<Self::Output>)?.into())
    }
    fn matmul_<U>(&self, rhs: Tensor<B>, out: U) -> anyhow::Result<Self::Output>
        where U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>
    {
        Ok(matmul_with_out(self, &rhs, Some(out))?.into())
    }
}

impl<A, B> Matmul<&Tensor<B>>
    for Tensor<A>
    where
        A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
        B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
        <A as NormalOut<B>>::Output: CommonBounds
{
    type Output = Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    fn matmul(&self, rhs: &Tensor<B>) -> anyhow::Result<Self::Output> {
        Ok(matmul_with_out(self, &rhs, None::<Self::Output>)?.into())
    }

    fn matmul_<U>(&self, rhs: &Tensor<B>, out: U) -> anyhow::Result<Self::Output>
        where U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>
    {
        Ok(matmul_with_out(self, rhs, Some(out))?.into())
    }
}
