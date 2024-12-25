use tensor_common::slice::Slice;
use tensor_traits::CommonBounds;
use anyhow::Result;
use crate::tensor::Tensor;

impl<T> Tensor<T> where T: CommonBounds {
    /// Extracts a slice of the tensor based on the provided indices.
    ///
    /// This method creates a new tensor that represents a slice of the original tensor.
    /// It slices the tensor according to the specified indices and returns a new tensor
    /// without copying the underlying data, but instead adjusting the shape and strides.
    ///
    /// # Arguments
    ///
    /// * `index` - A reference to a slice of `Slice` structs that define how to slice the tensor along each axis.
    ///   The `Slice` type allows for specifying ranges, single elements, and other slicing options.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the sliced tensor as a new tensor. If any slicing error occurs
    /// (e.g., out-of-bounds access), an error message is returned.
    pub fn slice(&self, index: &[Slice]) -> Result<Tensor<T>> {
        Ok(self.inner.slice(index)?.into())
    }
}
