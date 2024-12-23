//! This crate provides a memory allocator for tensor

#![deny(missing_docs)]

mod allocator;
#[cfg(feature = "cuda")]
mod cuda_allocator;
mod strorage;
extern crate lru;

pub use allocator::clone_storage;
#[cfg(feature = "cuda")]
pub use cuda_allocator::clone_storage as cuda_clone_storage;
pub use allocator::CACHE;
#[cfg(feature = "cuda")]
pub use cuda_allocator::CUDA_CACHE;
pub use strorage::CPU_STORAGE;
#[cfg(feature = "cuda")]
pub use strorage::CUDA_STORAGE;
/// program will free all the memory before exit
#[allow(non_snake_case)]
#[ctor::dtor]
fn free_pools() {
    CACHE.clear();
    #[cfg(feature = "cuda")]
    CUDA_CACHE.lock().unwrap().clear();
}