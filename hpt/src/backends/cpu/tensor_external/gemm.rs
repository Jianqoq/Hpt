use std::borrow::{Borrow, BorrowMut};

use crate::Tensor;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_traits::ops::binary::Gemm;
use hpt_traits::tensor::CommonBounds;
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};

type GemmOutput<A, B, const DEVICE: usize, A2> =
    Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, A2>;

impl<A, B, A2, const DEVICE: usize> Gemm<Tensor<B, Cpu, DEVICE, A2>> for Tensor<A, Cpu, DEVICE, A2>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = GemmOutput<A, B, DEVICE, A2>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = GemmOutput<A, B, DEVICE, A2>;

    fn gemm(
        &self,
        rhs: Tensor<B, Cpu, DEVICE, A2>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
    ) -> Result<Self::Output, TensorError> {
        Ok(self
            .inner
            .gemm(
                rhs.inner.as_ref(),
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
            )?
            .into())
    }
    fn gemm_<U>(
        &self,
        rhs: Tensor<B, Cpu, DEVICE, A2>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
        out: U,
    ) -> Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(self
            .inner
            .gemm_(
                rhs.inner.as_ref(),
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                out,
            )?
            .into())
    }
}

impl<A, B, A2, const DEVICE: usize> Gemm<&Tensor<B, Cpu, DEVICE, A2>> for Tensor<A, Cpu, DEVICE, A2>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = GemmOutput<A, B, DEVICE, A2>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = GemmOutput<A, B, DEVICE, A2>;

    fn gemm(
        &self,
        rhs: &Tensor<B, Cpu, DEVICE, A2>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
    ) -> Result<Self::Output, TensorError> {
        Ok(self
            .inner
            .gemm(
                rhs.inner.as_ref(),
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
            )?
            .into())
    }

    fn gemm_<U>(
        &self,
        rhs: &Tensor<B, Cpu, DEVICE, A2>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
        out: U,
    ) -> Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(self
            .inner
            .gemm_(
                rhs.inner.as_ref(),
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                out,
            )?
            .into())
    }
}
