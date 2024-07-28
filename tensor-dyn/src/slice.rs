use tensor_common::{ layout::Layout, pointer::Pointer, slice::{ slice_process, Slice } };
use tensor_traits::tensor::{ CommonBounds, TensorInfo };
use anyhow::Result;

use crate::{ backend::{ Backend, TensorBackend }, tensor::Tensor, tensor_base::_Tensor };

pub trait SliceOps<T, U> where T: CommonBounds {
    // slice operation mostly change the shape of tensor only
    fn slice(&self, ranges: U) -> Result<Self> where Self: Sized;
}

impl<T> _Tensor<T> where T: CommonBounds {
    fn slice_process(&self, index: &[Slice]) -> Result<_Tensor<T>> {
        let (res_shape, res_strides, offset) = slice_process(
            self.shape().to_vec(),
            self.strides().to_vec(),
            index,
            1
        )?;
        let res_ptr: *mut T = unsafe { self.data.ptr.offset(offset as isize) };
        return Ok(self.from_slice(res_ptr, res_shape, res_strides));
    }

    pub fn from_slice(&self, ptr: *mut T, shape: Vec<i64>, strides: Vec<i64>) -> _Tensor<T> {
        if self.parent.is_none() {
            return Self {
                data: Pointer::new(ptr),
                parent: Some(self.data),
                mem_layout: self.mem_layout.clone(),
                layout: Layout::new(shape, strides),
                _backend: Backend::new(),
            };
        } else {
            return Self {
                data: Pointer::new(ptr),
                parent: self.parent,
                mem_layout: self.mem_layout.clone(),
                layout: Layout::new(shape, strides),
                _backend: Backend::new(),
            };
        }
    }
}

impl<T, const N: usize> SliceOps<T, [Slice; N]> for _Tensor<T> where T: CommonBounds {
    fn slice(&self, slices: [Slice; N]) -> Result<_Tensor<T>> {
        self.slice_process(&slices)
    }
}

impl<T, const N: usize> SliceOps<T, &[Slice; N]> for _Tensor<T> where T: CommonBounds {
    fn slice(&self, slices: &[Slice; N]) -> Result<_Tensor<T>> {
        self.slice_process(slices)
    }
}

impl<T> SliceOps<T, &[Slice]> for _Tensor<T> where T: CommonBounds {
    fn slice(&self, slices: &[Slice]) -> Result<_Tensor<T>> {
        self.slice_process(slices)
    }
}

impl<T> SliceOps<T, &Vec<Slice>> for _Tensor<T> where T: CommonBounds {
    fn slice(&self, slices: &Vec<Slice>) -> Result<_Tensor<T>> {
        self.slice_process(slices)
    }
}

impl<T> Tensor<T> where T: CommonBounds {
    fn slice_process(&self, index: &[Slice]) -> Result<Tensor<T>> {
        Ok(self.inner.slice_process(index)?.into())
    }

    pub fn from_slice(&self, ptr: *mut T, shape: Vec<i64>, strides: Vec<i64>) -> Tensor<T> {
        self.inner.from_slice(ptr, shape, strides).into()
    }
}

impl<T, const N: usize> SliceOps<T, [Slice; N]> for Tensor<T> where T: CommonBounds {
    fn slice(&self, slices: [Slice; N]) -> Result<Tensor<T>> {
        self.slice_process(&slices)
    }
}

impl<T, const N: usize> SliceOps<T, &[Slice; N]> for Tensor<T> where T: CommonBounds {
    fn slice(&self, slices: &[Slice; N]) -> Result<Tensor<T>> {
        self.slice_process(slices)
    }
}

impl<T> SliceOps<T, &[Slice]> for Tensor<T> where T: CommonBounds {
    fn slice(&self, slices: &[Slice]) -> Result<Tensor<T>> {
        self.slice_process(slices)
    }
}

impl<T> SliceOps<T, &Vec<Slice>> for Tensor<T> where T: CommonBounds {
    fn slice(&self, slices: &Vec<Slice>) -> Result<Tensor<T>> {
        self.slice_process(slices)
    }
}
