use std::marker::PhantomData;

use hpt_allocator::traits::Allocator;
use hpt_common::{error::base::TensorError, layout::layout::Layout, slice::slice_process, Pointer};
use hpt_traits::{CommonBounds, Slice};

use crate::{tensor_base::_Tensor, BackendTy, Buffer, Tensor};

impl<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE: usize, A> Slice
    for Tensor<T, B, DEVICE, A>
where
    A: Allocator,
{
    fn slice(&self, index: &[(i64, i64, i64)]) -> Result<Tensor<T, B, DEVICE, A>, TensorError> {
        Ok(Tensor {
            inner: std::sync::Arc::new(self.inner.slice(index)?),
        })
    }
}

fn from_slice<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE: usize, A>(
    x: &_Tensor<T, B, DEVICE, A>,
    ptr: Pointer<T>,
    shape: Vec<i64>,
    strides: Vec<i64>,
) -> _Tensor<T, B, DEVICE, A>
where
    A: Allocator,
{
    // Create a new tensor, either as a child of a parent tensor or as a standalone tensor
    if x.parent.is_none() {
        let layout = Layout::new(shape, strides);
        _Tensor {
            data: ptr,
            parent: Some(x.data.clone()),
            mem_layout: x.mem_layout.clone(),
            layout,
            _backend: x._backend.clone(),
            phantom: PhantomData,
        }
    } else {
        let layout = Layout::new(shape, strides);
        _Tensor {
            data: ptr,
            parent: x.parent.clone(),
            mem_layout: x.mem_layout.clone(),
            layout,
            _backend: x._backend.clone(),
            phantom: PhantomData,
        }
    }
}

impl<T, B: BackendTy + Buffer + Clone, const DEVICE: usize, A> Slice for _Tensor<T, B, DEVICE, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn slice(
        &self,
        index: &[(i64, i64, i64)],
    ) -> std::result::Result<_Tensor<T, B, DEVICE, A>, TensorError> {
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
            Ok(from_slice(
                self,
                Pointer::new(res_ptr, len),
                res_shape,
                res_strides,
            ))
        }
        #[cfg(not(feature = "bound_check"))]
        {
            Ok(from_slice(
                self,
                Pointer::new(res_ptr),
                res_shape,
                res_strides,
            ))
        }
    }
}
