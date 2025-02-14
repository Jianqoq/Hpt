use std::{alloc::Layout, num::NonZeroUsize, sync::Mutex};

use crate::{ptr::SafePtr, storage::{cpu::CPU_STORAGE, CommonStorage, Storage}, traits::Allocator};
use hashbrown::{HashMap, HashSet};
use hpt_common::error::base::TensorError;
use lru::LruCache;
use once_cell::sync::Lazy;

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
    fn allocate(&mut self, layout: Layout, device_id: usize) -> Result<*mut u8, TensorError> {
        if let Some(allocator) = self.allocator.get_mut(&device_id) {
            allocator.allocate(layout, device_id)
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(NonZeroUsize::new(100).unwrap()),
                allocated: HashSet::new(),
            };
            let ptr = allocator.allocate(layout, device_id)?;
            self.allocator.insert(device_id, allocator);
            Ok(ptr)
        }
    }
    fn deallocate(&mut self, ptr: *mut u8, layout: &Layout, device_id: usize) {
        if let Some(allocator) = self.allocator.get_mut(&device_id) {
            allocator.deallocate(ptr, layout, device_id);
        } else {
            panic!("device {} not found in allocator", device_id);
        }
    }
    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        if let Some(allocator) = self.allocator.get_mut(&device_id) {
            allocator.insert_ptr(ptr, device_id);
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(NonZeroUsize::new(100).unwrap()),
                allocated: HashSet::new(),
            };
            allocator.insert_ptr(ptr, device_id);
            self.allocator.insert(device_id, allocator);
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
    fn allocate(
        &mut self,
        layout: Layout,
        device_id: usize,
    ) -> std::result::Result<*mut u8, TensorError> {
        if let Ok(mut storage) = CPU_STORAGE.lock() {
            crate::utils::allocate::allocate_helper(
                &mut self.cache,
                &mut self.allocated,
                &mut storage,
                || unsafe { std::alloc::alloc(layout) },
                |ptr, layout| unsafe { std::alloc::dealloc(ptr, layout) },
                layout,
                device_id,
            )
        } else {
            panic!("Failed to lock CPU_STORAGE");
        }
    }

    /// # Main Deallocation Function
    ///
    /// deallocate memory based on the ptr provided, if the ptr is found in the storage, decrement the reference count
    ///
    /// if the reference count is 0, remove the ptr from the storage, remove the ptr from the allocated set, and insert the ptr into the cache
    fn deallocate(&mut self, ptr: *mut u8, layout: &Layout, device_id: usize) {
        if let Ok(mut storage) = CPU_STORAGE.lock() {
            crate::utils::deallocate::deallocate_helper(
                &mut self.cache,
                &mut self.allocated,
                &mut storage,
                layout,
                ptr,
                device_id,
            );
        } else {
            panic!("Failed to lock CPU_STORAGE");
        }
    }

    /// # Insert Pointer
    ///
    /// insert the ptr into the allocated set, and increment the reference count in the storage
    ///
    /// this function is used to insert the ptr into the allocated set, and increment the reference count in the storage
    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        self.allocated.insert(SafePtr { ptr });
        if let Ok(mut map) = CPU_STORAGE.lock() {
            if let Some(storage) = map.get_mut(&device_id) {
                storage.increment_ref(SafePtr { ptr });
            } else {
                let mut storage = CommonStorage::new();
                storage.increment_ref(SafePtr { ptr });
                map.insert(device_id, storage);
            }
        }
    }
}
