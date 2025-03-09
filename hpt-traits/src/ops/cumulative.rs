use hpt_common::error::base::TensorError;

/// A trait for cumulative operations
pub trait CumulativeOps: Sized {
    /// Computes the cumulative sum of tensor elements along a specified axis.
    ///
    /// ## Parameters:
    /// `axis`: The axis along which to compute the cumulative sum
    /// - If specified: Compute cumsum along this axis
    /// - If None: Compute cumsum over the flattened array
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]);
    /// let cum_a = a.cumsum(None)?; // [1.0, 3.0, 6.0]
    ///
    /// let b = Tensor::<f32>::new(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let cum_b0 = b.cumsum(0)?;
    ///     // [[1.0, 2.0],
    ///     //  [4.0, 6.0]]
    /// ```
    #[track_caller]
    fn cumsum<A: Into<Option<i64>>>(&self, axis: A) -> Result<Self, TensorError>;
    /// Computes the cumulative product of tensor elements along a specified axis.
    ///
    /// ## Parameters:
    /// `axis`: The axis along which to compute the cumulative product
    /// - If specified: Compute cumprod along this axis
    /// - If None: Compute cumprod over the flattened array
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]);
    /// let cum_a = a.cumprod(None)?; // [1.0, 2.0, 6.0]
    ///
    /// let b = Tensor::<f32>::new(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let cum_b0 = b.cumprod(0)?;
    ///     // [[1.0, 2.0],
    ///     //  [3.0, 8.0]]
    /// ```
    #[track_caller]
    fn cumprod<A: Into<Option<i64>>>(&self, axis: A) -> Result<Self, TensorError>;
}
