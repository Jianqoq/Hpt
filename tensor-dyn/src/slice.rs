use tensor_common::{ layout::Layout, pointer::Pointer, slice::{ slice_process, Slice } };
use tensor_traits::tensor::{ CommonBounds, TensorInfo };
use anyhow::Result;

use crate::{ tensor::Tensor, tensor_base::_Tensor };

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
        Ok(self.from_slice(res_ptr, res_shape, res_strides))
    }

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
