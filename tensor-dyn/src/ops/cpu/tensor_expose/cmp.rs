use tensor_traits::{tensor::CommonBounds, TensorCmp};
use tensor_types::{
    dtype::TypeCommon,
    type_promote::{Cmp, SimdCmp},
};

use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use anyhow::Result;

impl<T> Tensor<T>
where
    T: CommonBounds,
{
    /// perform element-wise not equal operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    pub fn tensor_neq<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_neq(_rhs.inner.as_ref())?.into())
    }

    /// perform element-wise equal operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    pub fn tensor_eq<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.tensor_eq(_rhs.inner.as_ref())?.into())
    }

    /// perform element-wise less than operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    pub fn tensor_lt<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_lt(_rhs.inner.as_ref())?.into())
    }

    /// perform element-wise greater than operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    pub fn tensor_gt<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_gt(_rhs.inner.as_ref())?.into())
    }

    /// perform element-wise less than or equal operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    pub fn tensor_le<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_le(_rhs.inner.as_ref())?.into())
    }

    /// perform element-wise greater than or equal operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    pub fn tensor_ge<U: CommonBounds, D: Into<Tensor<U>>>(&self, rhs: D) -> Result<_Tensor<bool>>
    where
        T: Cmp<U>,
        <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec, Output = BoolVector>,
    {
        let _rhs: Tensor<U> = rhs.into();
        Ok(self.inner.as_ref().tensor_ge(_rhs.inner.as_ref())?.into())
    }
}
