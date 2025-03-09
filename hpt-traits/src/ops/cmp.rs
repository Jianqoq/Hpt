use std::borrow::Borrow;

use hpt_common::error::base::TensorError;

use crate::tensor::CommonBounds;

/// A trait for tensor comparison operations
pub trait TensorCmp<T: CommonBounds, C: CommonBounds> {
    /// right hand side tensor type
    type RHS;
    /// output tensor type, normally a boolean tensor
    type Output;

    /// check if element from x is not equal to element from y
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    /// let b = a.tensor_neq(&a)?; // [false false false]
    /// ```
    #[track_caller]
    fn tensor_neq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if element from x is equal to element from y
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    /// let b = a.tensor_eq(&a)?; // [true true true]
    /// ```
    #[track_caller]
    fn tensor_eq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if element from x is less than the element from y
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    /// let b = a.tensor_lt(&a)?; // [false false false]
    /// ```
    #[track_caller]
    fn tensor_lt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if element from x is greater than the element from y
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    /// let b = a.tensor_gt(&a)?; // [false false false]
    /// ```
    #[track_caller]
    fn tensor_gt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if element from x is less or equal to the element from y
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    /// let b = a.tensor_le(&a)?; // [true true true]
    /// ```
    #[track_caller]
    fn tensor_le<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;

    /// check if element from x is greater or equal to the element from y
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0, 2.0, 2.0]);
    /// let b = a.tensor_ge(&a)?; // [true true true]
    /// ```
    #[track_caller]
    fn tensor_ge<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>;
}
