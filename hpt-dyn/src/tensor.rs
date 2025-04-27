use hpt_common::{Pointer, layout::layout::Layout, shape::shape::Shape, strides::strides::Strides};
use std::sync::Arc;

use crate::DType;
use crate::utils::{backend::Backend, device::Device};

#[derive(Clone)]
pub struct Tensor {
    pub(crate) data: Pointer<u8>,
    pub(crate) layout: Layout,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) parent: Option<Pointer<u8>>,
    pub(crate) prg_update: Arc<dyn Fn(&mut [i64], &mut Pointer<u8>) + Send + Sync>,
    pub(crate) map_global_idx: Arc<dyn Fn(i64) -> i64>,
    pub(crate) map_gp: Arc<dyn Fn(i64) -> (i64, Vec<i64>)>,
    pub(crate) mem_layout: std::alloc::Layout,
    pub(crate) backend: Backend,
}

impl Tensor {
    pub fn as_slice<T: Sized>(&self) -> &[T] {
        if !self.is_contiguous() {
            panic!("uncontiguous tensor cannot be converted to slice");
        }
        unsafe {
            std::slice::from_raw_parts(
                self.data.ptr as *const T,
                self.mem_layout.size() as usize / std::mem::size_of::<T>(),
            )
        }
    }
    pub fn as_slice_mut<T: Sized>(&mut self) -> &mut [T] {
        if !self.is_contiguous() {
            panic!("uncontiguous tensor cannot be converted to slice");
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.ptr as *mut T,
                self.mem_layout.size() as usize / std::mem::size_of::<T>(),
            )
        }
    }
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
    pub fn size(&self) -> usize {
        self.layout.size() as usize
    }
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }
    pub fn strides(&self) -> &Strides {
        self.layout.strides()
    }
    pub fn ptr(&self) -> Pointer<u8> {
        self.data
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        self.backend.dealloc(self.mem_layout);
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

