use std::borrow::Borrow;

use anyhow::Result;
use tensor_types::{into_vec::IntoVec, type_promote::{Cmp, SimdCmp}};

use crate::CommonBounds;

/// A trait for tensor comparison operations
pub trait TensorCmp<T: CommonBounds> {
    /// right hand side tensor type
    type RHS<C>;
    /// output tensor type, normally a boolean tensor
    type Output;
    /// a boolean simd vector
    type BoolVector;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.eq(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_neq<C: CommonBounds, D>(&self, rhs: D) -> Result<Self::Output>
    where
        T: Cmp<C>,
        D: Borrow<Self::RHS<C>>,
        T::Vec: SimdCmp<C::Vec>,
        <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///     
    /// # Returns
    /// bool tensor
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.eq(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_eq<C: CommonBounds, D>(&self, rhs: D) -> Result<Self::Output>
    where
        T: Cmp<C>,
        D: Borrow<Self::RHS<C>>,
        T::Vec: SimdCmp<C::Vec>,
        <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.lt(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_lt<C: CommonBounds, D>(&self, rhs: D) -> Result<Self::Output>
    where
        T: Cmp<C>,
        D: Borrow<Self::RHS<C>>,
        T::Vec: SimdCmp<C::Vec>,
        <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.gt(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_gt<C: CommonBounds, D>(&self, rhs: D) -> Result<Self::Output>
    where
        T: Cmp<C>,
        D: Borrow<Self::RHS<C>>,
        T::Vec: SimdCmp<C::Vec>,
        <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.le(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_le<C: CommonBounds, D>(&self, rhs: D) -> Result<Self::Output>
    where
        T: Cmp<C>,
        D: Borrow<Self::RHS<C>>,
        T::Vec: SimdCmp<C::Vec>,
        <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.ge(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_ge<C: CommonBounds, D>(&self, rhs: D) -> Result<Self::Output>
    where
        T: Cmp<C>,
        D: Borrow<Self::RHS<C>>,
        T::Vec: SimdCmp<C::Vec>,
        <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;
}
