use hpt_common::error::base::TensorError;

/// trait for slicing tensor
pub trait Slice: Sized {
    /// Extracts a slice of the tensor based on the provided indices.
    ///
    /// This method creates a new tensor that represents a slice of the original tensor.
    /// It slices the tensor according to the specified indices and returns a new tensor
    /// without copying the underlying data, but instead adjusting the shape and strides.
    fn slice(&self, index: &[(i64, i64, i64)]) -> Result<Self, TensorError>;
}
