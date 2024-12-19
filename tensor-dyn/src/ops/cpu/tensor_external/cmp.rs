#![allow(unused)]

use std::borrow::Borrow;

use crate::ops::cpu::binary_normal::binary_fn_with_out_simd;
use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use anyhow::Result;
use tensor_traits::{tensor::CommonBounds, TensorCmp};
use tensor_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{Cmp, SimdCmp},
};

impl<T, C> TensorCmp<T, C> for Tensor<T>
where
    T: CommonBounds + Cmp<C, Output = bool>,
    C: CommonBounds,
    T::Vec: SimdCmp<C::Vec>,
    <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<BoolVector>,
{
    type RHS = Tensor<C>;
    type Output = Tensor<bool>;
    type BoolVector = BoolVector;
    /// perform element-wise not equal operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    fn tensor_neq<D>(&self, rhs: D) -> Result<Tensor<bool>>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_neq(rhs.borrow().inner.as_ref())?
            .into())
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
    fn tensor_eq<D>(&self, rhs: D) -> Result<Tensor<bool>>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_eq(rhs.borrow().inner.as_ref())?
            .into())
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
    fn tensor_lt<D>(&self, rhs: D) -> Result<Tensor<bool>>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_lt(rhs.borrow().inner.as_ref())?
            .into())
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
    fn tensor_gt<D>(&self, rhs: D) -> Result<Tensor<bool>>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_gt(rhs.borrow().inner.as_ref())?
            .into())
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
    fn tensor_le<D>(&self, rhs: D) -> Result<Tensor<bool>>
    where
        D: Borrow<Self::RHS>,
    {
        Ok(self
            .inner
            .as_ref()
            .tensor_le(rhs.borrow().inner.as_ref())?
            .into())
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
    fn tensor_ge<D>(&self, rhs: D) -> Result<Tensor<bool>>
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
