#![allow(unused)]

use std::borrow::Borrow;

use crate::backends::cpu::utils::binary::binary_normal::binary_fn_with_out_simd;
use crate::tensor::DiffTensor;
use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_traits::ops::cmp::TensorCmp;
use hpt_traits::tensor::CommonBounds;
use hpt_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{Cmp, SimdCmp},
};

impl<T, C, const DEVICE: usize, Al> TensorCmp<T, C> for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Cmp<C, Output = bool>,
    C: CommonBounds,
    T::Vec: SimdCmp<C::Vec>,
    <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<BoolVector>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type RHS = Tensor<C, Cpu, DEVICE, Al>;
    type Output = Tensor<bool, Cpu, DEVICE, Al>;

    fn tensor_neq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_neq(rhs.borrow().inner.as_ref())?
            .into())
    }

    fn tensor_eq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_eq(rhs.borrow().inner.as_ref())?
            .into())
    }

    fn tensor_lt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_lt(rhs.borrow().inner.as_ref())?
            .into())
    }

    fn tensor_gt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_gt(rhs.borrow().inner.as_ref())?
            .into())
    }

    fn tensor_le<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_le(rhs.borrow().inner.as_ref())?
            .into())
    }

    fn tensor_ge<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_ge(rhs.borrow().inner.as_ref())?
            .into())
    }
}

impl<T, C, const DEVICE: usize, Al> TensorCmp<T, C> for DiffTensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Cmp<C, Output = bool>,
    C: CommonBounds,
    T::Vec: SimdCmp<C::Vec>,
    <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<BoolVector>,
    Al: Allocator + 'static,
    Al::Output: AllocatorOutputRetrive,
{
    type RHS = DiffTensor<C, Cpu, DEVICE, Al>;
    type Output = Tensor<bool, Cpu, DEVICE, Al>;

    fn tensor_neq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        self.inner.tensor_neq(&rhs.borrow().inner)
    }

    fn tensor_eq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        self.inner.tensor_eq(&rhs.borrow().inner)
    }

    fn tensor_lt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        self.inner.tensor_lt(&rhs.borrow().inner)
    }

    fn tensor_gt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        self.inner.tensor_gt(&rhs.borrow().inner)
    }

    fn tensor_le<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        self.inner.tensor_le(&rhs.borrow().inner)
    }

    fn tensor_ge<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        self.inner.tensor_ge(&rhs.borrow().inner)
    }
}
