//! This crate provides a memory allocator for tensor

#![deny(missing_docs)]

mod allocators;
mod backend;
mod ptr;
mod storage;
pub(crate) mod utils {
    pub(crate) mod allocate;
    pub(crate) mod cache_resize;
    pub(crate) mod deallocate;
}
/// traits for the allocator
pub mod traits;

use std::{marker::PhantomData, sync::Arc};

pub use crate::allocators::cpu::CACHE;
#[cfg(feature = "cuda")]
pub use crate::allocators::cuda::CUDA_CACHE;
pub use crate::storage::clone_storage;
pub use allocators::cpu::resize_cpu_lru_cache;
#[cfg(feature = "cuda")]
pub use allocators::cuda::resize_cuda_lru_cache;
pub use backend::*;
pub use storage::cpu::CPU_STORAGE;
#[cfg(feature = "cuda")]
pub use storage::cuda::CUDA_STORAGE;
use traits::Allocator;

/// program will free all the memory before exit
#[allow(non_snake_case)]
#[ctor::dtor]
fn free_pools() {
    CACHE.lock().unwrap().clear();
    #[cfg(feature = "cuda")]
    CUDA_CACHE.lock().unwrap().clear();
}

/// Built-in allocator for Hpt
pub struct HptAllocator<B: BackendTy> {
    phantom: PhantomData<B>,
}

impl Allocator for HptAllocator<Cpu> {
    type Output = *mut u8;
    fn allocate(
        &mut self,
        layout: std::alloc::Layout,
        device_id: usize,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        CACHE.lock().unwrap().allocate(layout, device_id)
    }

    fn deallocate(&mut self, ptr: *mut u8, layout: &std::alloc::Layout, device_id: usize) {
        CACHE.lock().unwrap().deallocate(ptr, layout, device_id);
    }

    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        CACHE.lock().unwrap().insert_ptr(ptr, device_id);
    }

    fn clear(&mut self) {
        CACHE.lock().unwrap().clear();
    }
    
    fn new() -> Self {
        HptAllocator {
            phantom: PhantomData,
        }
    }
}

#[cfg(feature = "cuda")]
impl Allocator for HptAllocator<Cuda> {
    type Output = (*mut u8, Arc<cudarc::driver::CudaDevice>);

    fn allocate(
        &mut self,
        layout: std::alloc::Layout,
        device_id: usize,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        CUDA_CACHE.lock().unwrap().allocate(layout, device_id)
    }

    fn deallocate(&mut self, ptr: *mut u8, layout: &std::alloc::Layout, device_id: usize) {
        CUDA_CACHE
            .lock()
            .unwrap()
            .deallocate(ptr, layout, device_id);
    }

    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        CUDA_CACHE.lock().unwrap().insert_ptr(ptr, device_id);
    }

    fn clear(&mut self) {
        CUDA_CACHE.lock().unwrap().clear();
    }
    
    fn new() -> Self {
        HptAllocator {
            phantom: PhantomData,
        }
    }
}
