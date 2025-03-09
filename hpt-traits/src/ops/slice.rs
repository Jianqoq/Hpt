use hpt_common::error::base::TensorError;

/// trait for slicing tensor
pub trait Slice: Sized {
    /// Create a new Tensor by slicing an existing Tensor. Slicing allows you to extract a portion of a tensor using index ranges for each dimension.
    ///
    /// ## Parameters:
    /// `index`: `(start, end, step)`: Select from start to end with step
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::arange(0, 16)?.reshape(&[4, 4])?;
    /// let b = a.slice(&[(1, 3, 1), (0, 4, 1)])?;
    /// ```
    fn slice(&self, index: &[(i64, i64, i64)]) -> Result<Self, TensorError>;
}
