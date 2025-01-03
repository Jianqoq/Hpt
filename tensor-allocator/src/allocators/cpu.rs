use std::{alloc::Layout, num::NonZeroUsize, panic::Location, sync::Mutex};

use hashbrown::{HashMap, HashSet};
use lru::LruCache;
use once_cell::sync::Lazy;
use tensor_common::error::base::TensorError;
use tensor_common::error::memory::MemoryError;
use crate::{ptr::SafePtr, storage::Storage, storage::cpu::CPU_STORAGE, traits::Allocator};

/// `lru` cache allocator
pub static CACHE: Lazy<Mutex<CpuAllocator>> = Lazy::new(|| Mutex::new(CpuAllocator::new()));

/// # Allocator
///
/// a `lru` based allocator, to allocate and deallocate memory
///
/// this allocator is used widely in the library, to allocate and deallocate memory
///
/// # Safety
///
/// thread safe
///
/// # Potential Memory Leak
///
/// developer must carefully manage the reference count of the pointer allocated
pub struct CpuAllocator {
    allocator: HashMap<usize, _Allocator>,
}

impl Allocator for CpuAllocator {
    fn allocate(
        &mut self,
        layout: Layout,
        device_id: usize,
    ) -> Result<*mut u8, TensorError> {
        if let Some(allocator) = self.allocator.get_mut(&device_id) {
            allocator.allocate(layout, device_id)
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(NonZeroUsize::new(10).unwrap()),
                allocated: HashSet::new(),
            };
            let ptr = allocator.allocate(layout, device_id)?;
            self.allocator.insert(device_id, allocator);
            Ok(ptr)
        }
    }
    fn deallocate(&mut self, ptr: *mut u8, layout: &Layout, device_id: usize) {
        if let Some(allocator) = self.allocator.get_mut(&device_id) {
            allocator.deallocate(ptr, layout);
        }
    }

    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        if let Some(allocator) = self.allocator.get_mut(&device_id) {
            allocator.insert_ptr(ptr);
        }
    }

    fn clear(&mut self) {
        for (_, allocator) in self.allocator.iter_mut() {
            for (layout, ptrs) in allocator.cache.iter_mut() {
                for ptr in ptrs.iter() {
                    unsafe {
                        std::alloc::dealloc(ptr.ptr, layout.clone());
                    }
                }
            }
            allocator.cache.clear();
            assert_eq!(allocator.allocated.len(), 0);
        }
    }
}

impl CpuAllocator {
    pub fn new() -> Self {
        CpuAllocator {
            allocator: HashMap::new(),
        }
    }
}

struct _Allocator {
    cache: LruCache<Layout, Vec<SafePtr>>,
    allocated: HashSet<SafePtr>,
}

impl _Allocator {
    /// # Main Allocation Function
    /// allocating and freeing memory is expensive, we are using `LRU`(least recently used) algorithm to reuse the memory
    ///
    /// allocate memory based on layout provided, if the layout is not found in the cache, allocate, otherwise pop from the cache
    ///
    /// this function internally checks if the cache is full, if it is full, it pops the least recently used layout and deallocates the memory
    ///
    /// if the cache is not full, it inserts the allocated memory into the cache, and increments the reference count in the storage
    ///
    /// # Safety
    ///
    /// This function checks `null` ptr internally, any memory allocated through this method, downstream don't need to check for `null` ptr
    fn allocate(&mut self, layout: Layout, device_id: usize) -> std::result::Result<*mut u8, TensorError> {
        let ptr = if let Some(ptr) = self.cache.get_mut(&layout)
        /*check if we previously allocated same layout of memory */
        {
            // try pop the memory out, if it return None, there is no available cached memory, we need to allocate new memory
            if let Some(safe_ptr) = ptr.pop() {
                safe_ptr.ptr
            } else {
                let ptr = unsafe { std::alloc::alloc(layout) };
                if ptr.is_null() {
                    return Err(TensorError::Memory(
                        MemoryError::AllocationFailed {
                            device: "cpu".to_string(),
                            id: device_id,
                            size: layout.size() / 1024 / 1024,
                            location: Location::caller(),
                        })
                    );
                }
                self.allocated.insert(SafePtr { ptr });
                ptr
            }
        } else {
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(TensorError::Memory(
                    MemoryError::AllocationFailed {
                        device: "cpu".to_string(),
                        id: device_id,
                        size: layout.size() / 1024 / 1024,
                        location: Location::caller(),
                    })
                );
            }
            self.allocated.insert(SafePtr { ptr });
            ptr
        };
        // check if the cache is full, if it is full, pop the least recently used layout and deallocate the memory
        if self.cache.cap().get() == self.cache.len() {
            if let Some((layout, ptrs)) = self.cache.pop_lru() {
                for safe_ptr in ptrs {
                    unsafe {
                        std::alloc::dealloc(safe_ptr.ptr, layout);
                    }
                }
            }
        }
        // increment the reference count in the storage of the ptr allocated
        if let Ok(mut storage) = CPU_STORAGE.lock() {
            storage.increment_ref(SafePtr { ptr });
        }
        Ok(ptr)
    }

    /// # Main Deallocation Function
    ///
    /// deallocate memory based on the ptr provided, if the ptr is found in the storage, decrement the reference count
    ///
    /// if the reference count is 0, remove the ptr from the storage, remove the ptr from the allocated set, and insert the ptr into the cache
    fn deallocate(&mut self, ptr: *mut u8, layout: &Layout) {
        if let Ok(mut storage) = CPU_STORAGE.lock() {
            if storage.decrement_ref(SafePtr { ptr }) {
                self.allocated.remove(&SafePtr { ptr });
                if let Some(ptrs) = self.cache.get_mut(layout) {
                    ptrs.push(SafePtr { ptr });
                } else {
                    self.cache.put(layout.clone(), vec![SafePtr { ptr }]);
                }
            }
        }
    }

    /// # Insert Pointer
    ///
    /// insert the ptr into the allocated set, and increment the reference count in the storage
    ///
    /// this function is used to insert the ptr into the allocated set, and increment the reference count in the storage
    fn insert_ptr(&mut self, ptr: *mut u8) {
        self.allocated.insert(SafePtr { ptr });
        if let Ok(mut storage) = CPU_STORAGE.lock() {
            storage.increment_ref(SafePtr { ptr });
        }
    }
}
