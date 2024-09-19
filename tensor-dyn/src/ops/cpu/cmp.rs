#![allow(unused)]

use tensor_traits::tensor::CommonBounds;
use tensor_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{Cmp, SimdCmp},
};

use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use anyhow::Result;

use super::binary_normal::binary_fn_with_out_simd;

impl<T> _Tensor<T>
where
    T: CommonBounds,
{
    pub fn tensor_neq<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let _rhs = rhs.into();
        let res = binary_fn_with_out_simd(
            self,
            &_rhs,
            |x, y| x._ne(y),
            |x, y| x._ne(y).into_vec(),
            None::<_Tensor<bool>>,
        )?;
        Ok(res)
    }

    pub fn tensor_eq<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let _rhs = rhs.into();
        let res = binary_fn_with_out_simd(
            self,
            &_rhs,
            |x, y| x._eq(y),
            |x, y| x._eq(y).into_vec(),
            None::<_Tensor<bool>>,
        )?;
        Ok(res)
    }

    pub fn tensor_lt<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let _rhs = rhs.into();
        let res = binary_fn_with_out_simd(
            self,
            &_rhs,
            |x, y| x._lt(y),
            |x, y| x._lt(y).into_vec(),
            None::<_Tensor<bool>>,
        )?;
        Ok(res)
    }

    pub fn tensor_gt<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let _rhs: _Tensor<U> = rhs.into();
        let res = binary_fn_with_out_simd(
            self,
            &_rhs,
            |x, y| x._gt(y),
            |x, y| x._gt(y).into_vec(),
            None::<_Tensor<bool>>,
        )?;
        Ok(res)
    }

    pub fn tensor_le<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let _rhs: _Tensor<U> = rhs.into();
        let res = binary_fn_with_out_simd(
            self,
            &_rhs,
            |x, y| x._le(y),
            |x, y| x._le(y).into_vec(),
            None::<_Tensor<bool>>,
        )?;
        Ok(res)
    }

    pub fn tensor_ge<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let _rhs: _Tensor<U> = rhs.into();
        let res = binary_fn_with_out_simd(
            self,
            &_rhs,
            |x, y| x._ge(y),
            |x, y| x._ge(y).into_vec(),
            None::<_Tensor<bool>>,
        )?;
        Ok(res)
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds,
{
    pub fn tensor_neq<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_neq(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_eq<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.tensor_eq(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_lt<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_lt(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_gt<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_gt(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_le<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_le(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_ge<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_ge(_rhs.inner.as_ref())?.into())
    }
}
