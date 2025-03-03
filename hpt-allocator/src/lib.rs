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

use std::marker::PhantomData;

pub use crate::allocators::cpu::CACHE;
#[cfg(feature = "cuda")]
pub use crate::allocators::cuda::CUDA_CACHE;
pub use crate::storage::clone_storage;
pub use allocators::cpu::resize_cpu_lru_cache;
#[cfg(feature = "cuda")]
pub use allocators::cuda::resize_cuda_lru_cache;
pub use storage::cpu::CPU_STORAGE;
#[cfg(feature = "cuda")]
pub use storage::cuda::CUDA_STORAGE;
pub use backend::*;
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



