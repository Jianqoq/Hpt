//! Slice operations for dynamic tensors.

use crate::BackendTy;
use crate::{tensor_base::_Tensor, Buffer};
use tensor_common::error::base::TensorError;
use tensor_common::{
    layout::layout::Layout,
    slice::{slice_process, Slice},
    utils::pointer::Pointer,
};
use tensor_traits::tensor::CommonBounds;

impl<T, B: BackendTy + Buffer + Clone, const DEVICE: usize> _Tensor<T, B, DEVICE>
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
    pub fn slice(
        &self,
        index: &[Slice],
    ) -> std::result::Result<_Tensor<T, B, DEVICE>, TensorError> {
        let (res_shape, res_strides, offset) = slice_process(
            self.layout.shape().to_vec(),
            self.layout.strides().to_vec(),
            index,
            1,
        )?;
        let res_ptr: *mut T = unsafe { self.data.ptr.offset(offset as isize) };
        #[cfg(feature = "bound_check")]
        {
            if offset < 0 || offset >= (self.data.len as i64) {
                panic!(
                    "index out of bounds, got offset: {}, origin shape: {}, origin strides: {}, slices: {:?}",
                    offset,
                    self.layout.shape(),
                    self.layout.strides(),
                    index
                );
            }
            let len = self.data.len - offset;
            Ok(self.from_slice(Pointer::new(res_ptr, len), res_shape, res_strides))
        }
        #[cfg(not(feature = "bound_check"))]
        {
            Ok(self.from_slice(Pointer::new(res_ptr), res_shape, res_strides))
        }
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
    fn from_slice(
        &self,
        ptr: Pointer<T>,
        shape: Vec<i64>,
        strides: Vec<i64>,
    ) -> _Tensor<T, B, DEVICE> {
        // Create a new tensor, either as a child of a parent tensor or as a standalone tensor
        if self.parent.is_none() {
            let layout = Layout::new(shape, strides);
            Self {
                data: ptr,
                parent: Some(self.data.clone()),
                mem_layout: self.mem_layout.clone(),
                layout,
                _backend: self._backend.clone(),
            }
        } else {
            let layout = Layout::new(shape, strides);
            Self {
                data: ptr,
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                layout,
                _backend: self._backend.clone(),
            }
        }
    }
}
