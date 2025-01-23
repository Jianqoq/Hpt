#![allow(unused)]

use std::borrow::Borrow;

use crate::ops::cpu::utils::binary::binary_normal::binary_fn_with_out_simd;
use crate::Cpu;
use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use tensor_common::error::base::TensorError;
use tensor_traits::{tensor::CommonBounds, TensorCmp};
use tensor_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{Cmp, SimdCmp},
};

impl<T, C, const DEVICE: usize> TensorCmp<T, C> for _Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Cmp<C, Output = bool>,
    C: CommonBounds,
    T::Vec: SimdCmp<C::Vec>,
    <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<BoolVector>,
{
    type RHS = _Tensor<C, Cpu, DEVICE>;
    type Output = _Tensor<bool, Cpu, DEVICE>;
    type BoolVector = BoolVector;

    fn tensor_neq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._ne(y),
            |x, y| x._ne(y).into_vec(),
            None::<Self::Output>,
        )?;
        Ok(res)
    }

    fn tensor_eq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._eq(y),
            |x, y| x._eq(y).into_vec(),
            None::<Self::Output>,
        )?;
        Ok(res)
    }

    fn tensor_lt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._lt(y),
            |x, y| x._lt(y).into_vec(),
            None::<Self::Output>,
        )?;
        Ok(res)
    }

    fn tensor_gt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._gt(y),
            |x, y| x._gt(y).into_vec(),
            None::<Self::Output>,
        )?;
        Ok(res)
    }

    fn tensor_le<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._le(y),
            |x, y| x._le(y).into_vec(),
            None::<Self::Output>,
        )?;
        Ok(res)
    }

    fn tensor_ge<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._ge(y),
            |x, y| x._ge(y).into_vec(),
            None::<Self::Output>,
        )?;
        Ok(res)
    }
}
