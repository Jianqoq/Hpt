//! Slice operations for dynamic tensors.

use anyhow::Result;
use tensor_common::{
    layout::Layout,
    pointer::Pointer,
    slice::{slice_process, Slice},
};
use tensor_traits::tensor::{CommonBounds, TensorInfo};

use crate::tensor_base::_Tensor;

impl<T> _Tensor<T>
where
    T: CommonBounds,
{
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
    pub fn slice(&self, index: &[Slice]) -> Result<_Tensor<T>> {
        let (res_shape, res_strides, offset) =
            slice_process(self.shape().to_vec(), self.strides().to_vec(), index, 1)?;
        let res_ptr: *mut T = unsafe { self.data.ptr.offset(offset as isize) };
        Ok(self.from_slice(res_ptr, res_shape, res_strides))
    }
    /// Creates a new tensor from a slice of memory.
    ///
    /// This method is used internally to create a new tensor from a pointer to a memory region,
    /// along with the provided shape and strides. It manages the memory layout and ensures
    /// that the resulting tensor references the correct portion of the original tensor.
    ///
    /// # Arguments
    ///
    /// * `ptr` - A pointer to the memory region that the new tensor will reference.
    /// * `shape` - A vector representing the shape of the new tensor.
    /// * `strides` - A vector representing the strides of the new tensor, which determine
    ///   how the tensor's data is laid out in memory.
    ///
    /// # Returns
    ///
    /// Returns a new `_Tensor` referencing the specified slice of memory.
    pub fn from_slice(&self, ptr: *mut T, shape: Vec<i64>, strides: Vec<i64>) -> _Tensor<T> {
        let (shape, strides) = if shape.contains(&0) {
            let mut new_shape = Vec::new();
            let mut new_strides = Vec::new();
            for (i, &s) in shape.iter().enumerate() {
                if s == 0 {
                    continue;
                }
                new_shape.push(s);
                new_strides.push(strides[i]);
            }
            (new_shape, new_strides)
        } else {
            (shape, strides)
        };
        // Create a new tensor, either as a child of a parent tensor or as a standalone tensor
        if self.parent.is_none() {
            let layout = Layout::new(shape, strides);
            Self {
                #[cfg(feature = "bound_check")]
                data: Pointer::new(ptr, layout.clone()),
                #[cfg(not(feature = "bound_check"))]
                data: Pointer::new(ptr),
                parent: Some(self.data.clone()),
                mem_layout: self.mem_layout.clone(),
                layout,
                _backend: self._backend.clone(),
            }
        } else {
            let layout = Layout::new(shape, strides);
            Self {
                #[cfg(feature = "bound_check")]
                data: Pointer::new(ptr, layout.clone()),
                #[cfg(not(feature = "bound_check"))]
                data: Pointer::new(ptr),
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                layout,
                _backend: self._backend.clone(),
            }
        }
    }
}
