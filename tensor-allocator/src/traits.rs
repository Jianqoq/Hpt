use std::alloc::Layout;

use tensor_common::err_handler::ErrHandler;

/// traits for the allocator
pub trait Allocator {
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
    fn allocate(&mut self, layout: Layout, device_id: usize) -> Result<*mut u8, ErrHandler>;
    /// deallocate memory by using lru cache strategy
    ///
    /// # Logic
    ///
    /// 1. check if the ptr is found in the storage
    ///
    /// 2. if the ptr is found in the storage, decrement the reference count
    ///
    /// 3. if the reference count is 0, remove the ptr from the storage, remove the ptr from the allocated set, and insert the ptr into the cache
    fn deallocate(&mut self, ptr: *mut u8, layout: &Layout, device_id: usize);
    /// if the ptr is found in the storage, increment the reference count, otherwise insert the ptr into the storage
    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize);
    /// clear the cache, deallocate all the memory allocated
    ///
    /// this is used when the program exits, it will be called automatically
    fn clear(&mut self);
}
