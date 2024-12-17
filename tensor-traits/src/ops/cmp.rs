use std::borrow::Borrow;

use anyhow::Result;
use tensor_types::{ into_vec::IntoVec, type_promote::{ Cmp, SimdCmp } };

use crate::CommonBounds;

/// A trait for tensor comparison operations
pub trait TensorCmp<T: CommonBounds, C: CommonBounds> {
    /// right hand side tensor type
    type RHS;
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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_neq<D>(&self, rhs: D) -> Result<Self::Output>
        where
            T: Cmp<C>,
            D: Borrow<Self::RHS>,
            T::Vec: SimdCmp<C::Vec>,
            <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_eq<D>(&self, rhs: D) -> Result<Self::Output>
        where
            T: Cmp<C>,
            D: Borrow<Self::RHS>,
            T::Vec: SimdCmp<C::Vec>,
            <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_lt<D>(&self, rhs: D) -> Result<Self::Output>
        where
            T: Cmp<C>,
            D: Borrow<Self::RHS>,
            T::Vec: SimdCmp<C::Vec>,
            <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_gt<D>(&self, rhs: D) -> Result<Self::Output>
        where
            T: Cmp<C>,
            D: Borrow<Self::RHS>,
            T::Vec: SimdCmp<C::Vec>,
            <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_le<D>(&self, rhs: D) -> Result<Self::Output>
        where
            T: Cmp<C>,
            D: Borrow<Self::RHS>,
            T::Vec: SimdCmp<C::Vec>,
            <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tensor_ge<D>(&self, rhs: D) -> Result<Self::Output>
        where
            T: Cmp<C>,
            D: Borrow<Self::RHS>,
            T::Vec: SimdCmp<C::Vec>,
            <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<Self::BoolVector>;
}
