use hpt_common::error::base::TensorError;
use hpt_common::{ Pointer, layout::layout::Layout, shape::shape::Shape, strides::strides::Strides };
use hpt_traits::tensor::{ CommonBounds, TensorInfo };
use std::sync::Arc;

use crate::utils::index_cal::{
    dispatch_loop_progress_update,
    dispatch_map_global_idx,
    dispatch_map_gp,
};
use crate::{ DType, ALIGN };
use crate::utils::{ backend::Backend, device::Device };

use hpt_iterator::TensorIterator;

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
                (self.mem_layout.size() as usize) / std::mem::size_of::<T>()
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
                (self.mem_layout.size() as usize) / std::mem::size_of::<T>()
            )
        }
    }
    pub unsafe fn from_raw(
        data: *mut u8,
        layout: Layout,
        dtype: DType,
        device: Device
    ) -> Result<Self, TensorError> {
        let len = layout.size() as usize;
        match device {
            Device::Cpu => {
                let ptr = Pointer::new(data, len as i64);
                let prg_update = dispatch_loop_progress_update(&layout, dtype.sizeof());
                let map_global_idx = dispatch_map_global_idx(&layout);
                let map_gp = dispatch_map_gp(&layout);
                let mem_layout = std::alloc::Layout
                    ::from_size_align(len * dtype.sizeof(), ALIGN)
                    .expect("failed to create memory layout");
                Ok(Self {
                    data: ptr,
                    layout,
                    dtype,
                    device,
                    parent: None,
                    prg_update,
                    map_global_idx,
                    map_gp,
                    mem_layout,
                    backend: Backend::new_cpu(ptr, 0, false),
                })
            }
            #[cfg(feature = "cuda")]
            _ => { unimplemented!() }
        }
    }
}

macro_rules! impl_tensor_info {
    ($t:ty) => {
        impl TensorInfo for $t {
            fn ptr<T>(&self) -> Pointer<T> {
                self.data.cast::<T>()
            }
        
            fn size(&self) -> usize {
                self.layout.size() as usize
            }
        
            fn shape(&self) -> &Shape {
                self.layout.shape()
            }
        
            fn strides(&self) -> &Strides {
                self.layout.strides()
            }
        
            fn layout(&self) -> &Layout {
                &self.layout
            }
        
            fn parent<T>(&self) -> Option<Pointer<T>> {
                self.parent.map(|p| p.cast::<T>())
            }
        
            fn ndim(&self) -> usize {
                self.layout.ndim()
            }
        
            fn is_contiguous(&self) -> bool {
                self.layout.is_contiguous()
            }
        }
    };
}

impl_tensor_info!(Tensor);
impl_tensor_info!(&Tensor);
impl_tensor_info!(&mut Tensor);

impl<T: CommonBounds> TensorIterator<'_, T> for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        self.backend.dealloc(self.mem_layout);
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}
