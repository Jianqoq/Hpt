use std::borrow::Borrow;

use hpt_common::error::base::TensorError;

use crate::CommonBounds;

/// A trait for tensor comparison operations
pub trait TensorCmp<T: CommonBounds, C: CommonBounds> {
    /// right hand side tensor type
    type RHS;
    /// output tensor type, normally a boolean tensor
    type Output;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[track_caller]
    fn tensor_neq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[track_caller]
    fn tensor_eq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[track_caller]
    fn tensor_lt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[track_caller]
    fn tensor_gt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[track_caller]
    fn tensor_le<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if two tensors are equal, return a bool tensor
    ///
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///
    /// # Returns
    /// bool tensor
    #[track_caller]
    fn tensor_ge<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;
}
