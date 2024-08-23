use tensor_traits::{ ops::cmp::TensorCmp, tensor::CommonBounds };
use tensor_types::type_promote::Cmp;
use anyhow::Result;
use crate::{ tensor::Tensor, tensor_base::_Tensor };

use super::binary_normal::binary_fn;

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
