mod allocator;
mod strorage;
extern crate lru;

pub use strorage::WGPU_STORAGE;
pub use strorage::CPU_STORAGE;
pub use allocator::CACHE;
pub use allocator::WGPU_CACHE;
pub use allocator::DeviceWrapper;
pub use allocator::WgpuAllocator;
pub use allocator::BufferWrapper;
pub use allocator::clone_storage;

// #[dtor]
// fn free_pools() {
//     unsafe {
//         CACHE.clear();
//     }
// }
