use hpt_common::error::base::TensorError;

/// A trait for cumulative operations
pub trait CumulativeOps: Sized {
    /// Computes the cumulative sum of the elements along a specified axis.
    ///
    /// This method calculates the cumulative sum of the elements in the tensor along the given `axis`.
    /// The cumulative sum of an element at position `i` is the sum of all elements from the start of the axis
    /// up to and including position `i`. If no axis is specified, the cumulative sum is computed over a flattened
    /// version of the tensor.
    ///
    /// # Arguments
    ///
    /// * `axis` - An optional axis along which to compute the cumulative sum. If `None`, the tensor is flattened,
    ///   and the cumulative sum is computed over all elements.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a new tensor with the cumulative sum computed along the specified axis.
    #[track_caller]
    fn cumsum<A: Into<Option<i64>>>(&self, axis: A) -> Result<Self, TensorError>;
    /// Computes the cumulative product of the elements along a specified axis.
    ///
    /// This method calculates the cumulative product of the elements in the tensor along the given `axis`.
    /// The cumulative product of an element at position `i` is the product of all elements from the start of the axis
    /// up to and including position `i`. If no axis is specified, the cumulative product is computed over a flattened
    /// version of the tensor.
    ///
    /// # Arguments
    ///
    /// * `axis` - An optional axis along which to compute the cumulative product. If `None`, the tensor is flattened,
    ///   and the cumulative product is computed over all elements.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a new tensor with the cumulative product computed along the specified axis.
    #[track_caller]
    fn cumprod<A: Into<Option<i64>>>(&self, axis: A) -> Result<Self, TensorError>;
}
