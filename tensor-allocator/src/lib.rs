//! This crate provides a memory allocator for tensor

#![deny(missing_docs)]

mod allocators;
mod storage;
/// traits for the allocator
pub mod traits;
mod ptr;

#[cfg(feature = "cuda")]
mod cuda_allocator;
extern crate lru;

#[cfg(feature = "cuda")]
pub use cuda_allocator::clone_storage as cuda_clone_storage;
#[cfg(feature = "cuda")]
pub use cuda_allocator::CUDA_CACHE;
pub use storage::cpu::CPU_STORAGE;
#[cfg(feature = "cuda")]
pub use storage::cuda::CUDA_STORAGE;
use traits::Allocator;
pub use crate::allocators::cpu::CACHE;
pub use crate::storage::cpu::clone_storage;

/// program will free all the memory before exit
#[allow(non_snake_case)]
#[ctor::dtor]
fn free_pools() {
    CACHE.lock().unwrap().clear();
    #[cfg(feature = "cuda")]
    CUDA_CACHE.lock().unwrap().clear();
}