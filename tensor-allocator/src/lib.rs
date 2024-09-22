//! This crate provides a memory allocator for tensor

#![deny(missing_docs)]

mod allocator;
mod strorage;
extern crate lru;

pub use allocator::clone_storage;
pub use allocator::CACHE;
pub use strorage::CPU_STORAGE;

/// program will free all the memory before exit
#[allow(non_snake_case)]
#[ctor::dtor]
fn free_pools() {
    CACHE.clear();
}
