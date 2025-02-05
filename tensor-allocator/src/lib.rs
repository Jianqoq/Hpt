//! This crate provides a memory allocator for tensor

#![deny(missing_docs)]

mod allocators;
mod ptr;
mod storage;
/// traits for the allocator
pub mod traits;

extern crate lru;

pub use crate::allocators::cpu::CACHE;
pub use crate::storage::cpu::clone_storage;
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
