use std::alloc::Layout;

use hpt_common::error::base::TensorError;

/// traits for the allocator
pub trait Allocator: Clone {
    /// the output type of the allocator
    type Output: AllocatorOutputRetrive;
    /// cpu type of the allocator
    type CpuAllocator: Allocator;
    /// cuda type of the allocator
    #[cfg(feature = "cuda")]
    type CudaAllocator: Allocator;
    /// allocate memory by using lru cache strategy
    ///
    /// # Logic
    ///
    /// 1. check if the layout is found in the cache
    ///
    /// 2. if the layout is found in the cache, pop the memory out, if it return None, there is no available cached memory, we need to allocate new memory
    ///
    /// 3. if the layout is not found in the cache, allocate new memory
    ///
    /// 4. eventually, if the cache is full, pop the least recently used memory and deallocate the memory
    fn allocate(&self, layout: Layout, device_id: usize) -> Result<Self::Output, TensorError>;

    /// similar to `allocate`, but the memory is zeroed
    fn allocate_zeroed(
        &self,
        layout: Layout,
        device_id: usize,
    ) -> Result<Self::Output, TensorError>;

    /// deallocate memory by using lru cache strategy
    ///
    /// # Logic
    ///
    /// 1. check if the ptr is found in the storage
    ///
    /// 2. if the ptr is found in the storage, decrement the reference count
    ///
    /// 3. if the reference count is 0, remove the ptr from the storage, remove the ptr from the allocated set, and insert the ptr into the cache
    fn deallocate(&self, ptr: *mut u8, layout: &Layout, should_drop: bool, device_id: usize);
    /// if the ptr is found in the storage, increment the reference count, otherwise insert the ptr into the storage
    fn insert_ptr(&self, ptr: *mut u8, device_id: usize);
    /// clear the cache, deallocate all the memory allocated
    ///
    /// this is used when the program exits, it will be called automatically
    fn clear(&self);

    /// forget the data in the allocator
    fn forget(&self, ptr: *mut u8, device_id: usize);

    /// create a new allocator
    fn new() -> Self;
}

/// traits for the allocator output retrive
pub trait AllocatorOutputRetrive {
    /// get the pointer from the allocator output
    fn get_ptr(&self) -> *mut u8;
    /// get the device from the allocator output
    #[cfg(feature = "cuda")]
    fn get_device(&self) -> std::sync::Arc<cudarc::driver::CudaDevice>;
}

impl AllocatorOutputRetrive for *mut u8 {
    fn get_ptr(&self) -> *mut u8 {
        self.clone()
    }
    #[cfg(feature = "cuda")]
    fn get_device(&self) -> std::sync::Arc<cudarc::driver::CudaDevice> {
        panic!("cuda is not enabled");
    }
}

#[cfg(feature = "cuda")]
impl AllocatorOutputRetrive for (*mut u8, std::sync::Arc<cudarc::driver::CudaDevice>) {
    fn get_ptr(&self) -> *mut u8 {
        self.0.clone()
    }
    #[cfg(feature = "cuda")]
    fn get_device(&self) -> std::sync::Arc<cudarc::driver::CudaDevice> {
        self.1.clone()
    }
}
/// traits for the allocator output convert to backend
pub trait FromAllocatorOutput<T> {
    /// convert the allocator output to backend
    fn from_allocator_output(alloc_output: T) -> Self;
}
