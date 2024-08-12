mod allocator;
mod strorage;
extern crate lru;

use ctor::dtor;
pub use strorage::WGPU_STORAGE;
pub use strorage::CPU_STORAGE;
pub use allocator::CACHE;
pub use allocator::WGPU_CACHE;
pub use allocator::DeviceWrapper;
pub use allocator::WgpuAllocator;
pub use allocator::BufferWrapper;

// #[dtor]
// fn free_pools() {
//     unsafe {
//         CACHE.clear();
//     }
// }