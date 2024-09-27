#![allow(unused)]

use std::borrow::Borrow;

use tensor_traits::{ tensor::CommonBounds, TensorCmp };
use tensor_types::{ dtype::TypeCommon, into_vec::IntoVec, type_promote::{ Cmp, SimdCmp } };
use crate::ops::cpu::binary_normal::binary_fn_with_out_simd;
use crate::{ tensor::Tensor, tensor_base::_Tensor, BoolVector };
use anyhow::Result;

impl<T> TensorCmp<T> for _Tensor<T> where T: CommonBounds {
    type Output = _Tensor<bool>;
    type RHS<C> = _Tensor<C>;
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
    fn tensor_neq<C, D>(&self, rhs: D) -> Result<_Tensor<bool>>
        where
            T: Cmp<C>,
            D: Borrow<_Tensor<C>>,
            C: CommonBounds,
            T::Vec: SimdCmp<C::Vec>,
            <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<BoolVector>
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._ne(y),
            |x, y| x._ne(y).into_vec(),
            None::<_Tensor<bool>>
        )?;
        Ok(res)
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
    fn tensor_eq<U: CommonBounds, D>(&self, rhs: D) -> Result<_Tensor<bool>>
        where
            T: Cmp<U>,
            D: Borrow<_Tensor<U>>,
            <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
            <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._eq(y),
            |x, y| x._eq(y).into_vec(),
            None::<_Tensor<bool>>
        )?;
        Ok(res)
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
    fn tensor_lt<U: CommonBounds, D>(&self, rhs: D) -> Result<_Tensor<bool>>
        where
            T: Cmp<U>,
            D: Borrow<_Tensor<U>>,
            <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
            <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._lt(y),
            |x, y| x._lt(y).into_vec(),
            None::<_Tensor<bool>>
        )?;
        Ok(res)
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
    fn tensor_gt<U: CommonBounds, D>(&self, rhs: D) -> Result<_Tensor<bool>>
        where
            T: Cmp<U>,
            D: Borrow<_Tensor<U>>,
            <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
            <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._gt(y),
            |x, y| x._gt(y).into_vec(),
            None::<_Tensor<bool>>
        )?;
        Ok(res)
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
    fn tensor_le<U: CommonBounds, D>(&self, rhs: D) -> Result<_Tensor<bool>>
        where
            T: Cmp<U>,
            D: Borrow<_Tensor<U>>,
            <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
            <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._le(y),
            |x, y| x._le(y).into_vec(),
            None::<_Tensor<bool>>
        )?;
        Ok(res)
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
    fn tensor_ge<U: CommonBounds, D>(&self, rhs: D) -> Result<_Tensor<bool>>
        where
            T: Cmp<U>,
            D: Borrow<_Tensor<U>>,
            <T as TypeCommon>::Vec: SimdCmp<<U as TypeCommon>::Vec>,
            <<T as TypeCommon>::Vec as SimdCmp<<U as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>
    {
        let res = binary_fn_with_out_simd(
            self,
            rhs.borrow(),
            |x, y| x._ge(y),
            |x, y| x._ge(y).into_vec(),
            None::<_Tensor<bool>>
        )?;
        Ok(res)
    }
}
