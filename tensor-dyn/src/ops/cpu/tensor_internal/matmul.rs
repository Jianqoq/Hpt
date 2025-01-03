use std::borrow::{ Borrow, BorrowMut };

use tensor_common::error::base::TensorError;
use tensor_traits::{ CommonBounds, Matmul };
use tensor_types::{ into_scalar::IntoScalar, type_promote::NormalOut };

use crate::{ ops::cpu::matmul::matmul_with_out, tensor_base::_Tensor };

impl<A, B> Matmul<_Tensor<B>>
    for _Tensor<A>
    where
        A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
        B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
        <A as NormalOut<B>>::Output: CommonBounds
{
    type Output = _Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    fn matmul(&self, rhs: _Tensor<B>) -> Result<Self::Output, TensorError> {
        matmul_with_out(self, &rhs, None::<Self::Output>)
    }
    fn matmul_<U>(&self, rhs: _Tensor<B>, out: U) -> Result<Self::Output, TensorError>
        where U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>
    {
        matmul_with_out(self, &rhs, Some(out))
    }
}

impl<A, B> Matmul<&_Tensor<B>>
    for _Tensor<A>
    where
        A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
        B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
        <A as NormalOut<B>>::Output: CommonBounds
{
    type Output = _Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    fn matmul(&self, rhs: &_Tensor<B>) -> Result<Self::Output, TensorError> {
        matmul_with_out(self, &rhs, None::<Self::Output>)
    }

    fn matmul_<U>(&self, rhs: &_Tensor<B>, out: U) -> Result<Self::Output, TensorError>
        where U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>
    {
        matmul_with_out(self, rhs, Some(out))
    }
}
