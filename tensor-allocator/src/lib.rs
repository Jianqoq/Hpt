mod allocator;
mod strorage;
extern crate lru;

pub use strorage::CPU_STORAGE;
pub use allocator::CACHE;
pub use allocator::clone_storage;

#[allow(non_snake_case)]
#[ctor::dtor]
fn free_pools() {
    unsafe {
        CACHE.clear();
    }
}
