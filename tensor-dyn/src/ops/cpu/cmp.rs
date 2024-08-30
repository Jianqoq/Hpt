#![allow(unused)]

use tensor_traits::tensor::CommonBounds;
use tensor_types::{ dtype::TypeCommon, into_vec::IntoVec, type_promote::{ Cmp, SimdCmp }, vectors::boolx32::boolx32 };
use anyhow::Result;
use crate::{ tensor::Tensor, tensor_base::_Tensor };

use super::binary_normal::{ binary_fn, binary_fn_simd };
#[cfg(not(feature = "simd"))]
impl<T, U> TensorCmp<T, U> for _Tensor<T> where T: CommonBounds, U: CommonBounds {
    type RHS = _Tensor<U>;

    type Output = _Tensor<bool>;

    fn neq<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: _Tensor<U> = rhs.into();
        let res: _Tensor<bool> = binary_fn(self, &_rhs, |x, y| x._ne(y))?;
        Ok(res)
    }

    fn eq<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: _Tensor<U> = rhs.into();
        let res: _Tensor<bool> = binary_fn(self, &_rhs, |x, y| x._eq(y))?;
        Ok(res)
    }

    fn lt<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: _Tensor<U> = rhs.into();
        let res: _Tensor<bool> = binary_fn(self, &_rhs, |x, y| x._lt(y))?;
        Ok(res)
    }

    fn gt<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: _Tensor<U> = rhs.into();
        let res: _Tensor<bool> = binary_fn(self, &_rhs, |x, y| x._gt(y))?;
        Ok(res)
    }

    fn le<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: _Tensor<U> = rhs.into();
        let res: _Tensor<bool> = binary_fn(self, &_rhs, |x, y| x._le(y))?;
        Ok(res)
    }

    fn ge<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: _Tensor<U> = rhs.into();
        let res: _Tensor<bool> = binary_fn(self, &_rhs, |x, y| x._ge(y))?;
        Ok(res)
    }
}

#[cfg(not(feature = "simd"))]
impl<T, U> TensorCmp<T, U> for Tensor<T> where T: CommonBounds, U: CommonBounds {
    type RHS = Tensor<U>;

    type Output = Tensor<bool>;

    fn neq<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().neq(_rhs.inner.as_ref())?.into())
    }

    fn eq<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: Tensor<U> = rhs.into();
        Ok(TensorCmp::<T, U>::eq(self.inner.as_ref(), _rhs.inner.as_ref())?.into())
    }

    fn lt<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().lt(_rhs.inner.as_ref())?.into())
    }

    fn gt<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().gt(_rhs.inner.as_ref())?.into())
    }

    fn le<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().le(_rhs.inner.as_ref())?.into())
    }

    fn ge<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output> where T: Cmp<U> {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().ge(_rhs.inner.as_ref())?.into())
    }
}

#[cfg(target_feature = "avx2")]
type BoolVec = boolx32;
#[cfg(target_feature = "avx512f")]
type BoolVec = boolx64;
#[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
type BoolVec = boolx16;

#[cfg(feature = "simd")]
impl<T> _Tensor<T> where T: CommonBounds {
    pub fn tensor_neq<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVec>
    {
        let _rhs = rhs.into();
        let res = binary_fn_simd(
            self,
            &_rhs,
            |x, y| x._ne(y),
            |x, y| x._ne(y).into_vec()
        )?;
        Ok(res)
    }

    pub fn tensor_eq<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVec>
    {
        let _rhs = rhs.into();
        let res = binary_fn_simd(
            self,
            &_rhs,
            |x, y| x._eq(y),
            |x, y| x._eq(y).into_vec()
        )?;
        Ok(res)
    }

    pub fn tensor_lt<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVec>
    {
        let _rhs = rhs.into();
        let res = binary_fn_simd(
            self,
            &_rhs,
            |x, y| x._lt(y),
            |x, y| x._lt(y).into_vec()
        )?;
        Ok(res)
    }

    pub fn tensor_gt<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVec>
    {
        let _rhs: _Tensor<U> = rhs.into();
        let res = binary_fn_simd(
            self,
            &_rhs,
            |x, y| x._gt(y),
            |x, y| x._gt(y).into_vec()
        )?;
        Ok(res)
    }

    pub fn tensor_le<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVec>
    {
        let _rhs: _Tensor<U> = rhs.into();
        let res = binary_fn_simd(
            self,
            &_rhs,
            |x, y| x._le(y),
            |x, y| x._le(y).into_vec()
        )?;
        Ok(res)
    }

    pub fn tensor_ge<U: CommonBounds, D: Into<_Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVec>
    {
        let _rhs: _Tensor<U> = rhs.into();
        let res = binary_fn_simd(
            self,
            &_rhs,
            |x, y| x._ge(y),
            |x, y| x._ge(y).into_vec()
        )?;
        Ok(res)
    }
}

impl<T> Tensor<T> where T: CommonBounds {
    pub fn tensor_neq<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVec>
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_neq(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_eq<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVec>
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.tensor_eq(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_lt<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVec>
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_lt(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_gt<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVec>
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_gt(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_le<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVec>
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_le(_rhs.inner.as_ref())?.into())
    }

    pub fn tensor_ge<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
        where T: Cmp<U>, <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVec>
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_ge(_rhs.inner.as_ref())?.into())
    }
}
