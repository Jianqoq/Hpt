use std::{
    alloc::Layout,
    collections::HashSet,
    num::NonZeroUsize,
    sync::atomic::{ AtomicUsize, Ordering },
};

use crate::{
    ptr::SafePtr,
    storage::{ cpu::CPU_STORAGE, CommonStorage, Storage },
    traits::Allocator,
};
use dashmap::DashMap;
use hpt_common::error::base::TensorError;
use lru::LruCache;
use once_cell::sync::Lazy;

/// `lru` cache allocator
pub(crate) static CACHE: Lazy<CpuAllocator> = Lazy::new(|| CpuAllocator::new());

pub(crate) static CPU_LRU_CACHE_SIZE: AtomicUsize = AtomicUsize::new(100);

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
#[derive(Clone)]
pub struct CpuAllocator {
    allocator: DashMap<usize, _Allocator>,
}

impl Allocator for CpuAllocator {
    type Output = *mut u8;
    type CpuAllocator = CpuAllocator;
    #[cfg(feature = "cuda")]
    type CudaAllocator = crate::allocators::cuda::CudaAllocator;
    fn allocate(&self, layout: Layout, device_id: usize) -> Result<Self::Output, TensorError> {
        if let Some(mut allocator) = self.allocator.get_mut(&device_id) {
            allocator.allocate(layout, device_id)
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(
                    NonZeroUsize::new(CPU_LRU_CACHE_SIZE.load(Ordering::Relaxed)).unwrap()
                ),
                allocated: HashSet::new(),
            };
            let ptr = allocator.allocate(layout, device_id)?;
            self.allocator.insert(device_id, allocator);
            Ok(ptr)
        }
    }
    fn allocate_zeroed(
        &self,
        layout: Layout,
        device_id: usize
    ) -> Result<Self::Output, TensorError> {
        if let Some(mut allocator) = self.allocator.get_mut(&device_id) {
            allocator.allocate_zeroed(layout, device_id)
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(
                    NonZeroUsize::new(CPU_LRU_CACHE_SIZE.load(Ordering::Relaxed)).unwrap()
                ),
                allocated: HashSet::new(),
            };
            let ptr = allocator.allocate_zeroed(layout, device_id)?;
            self.allocator.insert(device_id, allocator);
            Ok(ptr)
        }
    }

    fn deallocate(&self, ptr: *mut u8, layout: &Layout, should_drop: bool, device_id: usize) {
        if let Some(mut allocator) = self.allocator.get_mut(&device_id) {
            allocator.deallocate(ptr, layout, should_drop, device_id);
        } else {
            if !should_drop {
                return;
            }
            panic!("device {} not found in allocator", device_id);
        }
    }
    fn insert_ptr(&self, ptr: *mut u8, device_id: usize) {
        if let Some(mut allocator) = self.allocator.get_mut(&device_id) {
            allocator.insert_ptr(ptr, device_id);
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(
                    NonZeroUsize::new(CPU_LRU_CACHE_SIZE.load(Ordering::Relaxed)).unwrap()
                ),
                allocated: HashSet::new(),
            };
            allocator.insert_ptr(ptr, device_id);
            self.allocator.insert(device_id, allocator);
        }
    }
    fn clear(&self) {
        for mut allocators in self.allocator.iter_mut() {
            let device_id = *allocators.key();
            for (layout, ptrs) in allocators.cache.iter_mut() {
                for ptr in ptrs.iter() {
                    CPU_STORAGE.get_mut(&device_id)
                        .unwrap()
                        .decrement_ref(SafePtr { ptr: ptr.ptr });
                    unsafe {
                        std::alloc::dealloc(ptr.ptr, layout.clone());
                    }
                }
            }
            allocators.cache.clear();
            assert_eq!(allocators.allocated.len(), 0);
            assert_eq!(CPU_STORAGE.get(&device_id).unwrap().storage.len(), 0);
        }
    }

    fn new() -> Self {
        CpuAllocator {
            allocator: DashMap::new(),
        }
    }

    /// # Forget
    ///
    /// forget the ptr from the storage, remove the ptr from the allocated set
    fn forget(&self, ptr: *mut u8, device_id: usize) {
        if let Some(mut allocator) = self.allocator.get_mut(&device_id) {
            if allocator.allocated.get(&(SafePtr { ptr })).is_some() {
                allocator.allocated.remove(&(SafePtr { ptr }));
            }
        }
    }
}

#[derive(Clone)]
struct _Allocator {
    cache: LruCache<Layout, Vec<SafePtr>>,
    allocated: HashSet<SafePtr>,
}

impl _Allocator {
    fn allocate(&mut self, layout: Layout, device_id: usize) -> Result<*mut u8, TensorError> {
        crate::utils::allocate::allocate_helper(
            &mut self.cache,
            &mut self.allocated,
            &*CPU_STORAGE,
            || unsafe { std::alloc::alloc(layout) },
            |_, _| {},
            |ptr, layout| unsafe { std::alloc::dealloc(ptr, layout) },
            layout,
            device_id
        )
    }

    fn allocate_zeroed(
        &mut self,
        layout: Layout,
        device_id: usize
    ) -> Result<*mut u8, TensorError> {
        crate::utils::allocate::allocate_helper(
            &mut self.cache,
            &mut self.allocated,
            &*CPU_STORAGE,
            || unsafe { std::alloc::alloc_zeroed(layout) },
            |ptr, layout| unsafe {
                std::ptr::write_bytes(ptr, 0, layout.size());
            },
            |ptr, layout| unsafe { std::alloc::dealloc(ptr, layout) },
            layout,
            device_id
        )
    }

    /// # Main Deallocation Function
    ///
    /// deallocate memory based on the ptr provided, if the ptr is found in the storage, decrement the reference count
    ///
    /// if the reference count is 0, remove the ptr from the storage, remove the ptr from the allocated set, and insert the ptr into the cache
    fn deallocate(&mut self, ptr: *mut u8, layout: &Layout, should_drop: bool, device_id: usize) {
        crate::utils::deallocate::deallocate_helper(
            &mut self.cache,
            &mut self.allocated,
            &*CPU_STORAGE,
            layout,
            ptr,
            should_drop,
            device_id
        );
    }

    /// # Insert Pointer
    ///
    /// insert the ptr into the allocated set, and increment the reference count in the storage
    ///
    /// this function is used to insert the ptr into the allocated set, and increment the reference count in the storage
    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        self.allocated.insert(SafePtr { ptr });
        if let Some(mut storage) = CPU_STORAGE.get_mut(&device_id) {
            storage.increment_ref(SafePtr { ptr });
        } else {
            let mut storage = CommonStorage::new();
            storage.increment_ref(SafePtr { ptr });
            CPU_STORAGE.insert(device_id, storage);
        }
    }
}

/// resize the lru cache of the cpu allocator
///
/// when `new_size` >= `old_size`, cache size will increase and data won't be deallocated
///
/// when `new_size` < `old_size`, all the data in cache will be deallocated
pub fn resize_cpu_lru_cache(new_size: usize, device_id: usize) {
    if let Some(mut allocator) = CACHE.allocator.get_mut(&device_id) {
        crate::utils::cache_resize::resize_lru_cache(
            &mut allocator.cache,
            |ptr, layout| unsafe { std::alloc::dealloc(ptr, layout) },
            new_size
        );
    } else {
        let allocator = _Allocator {
            cache: LruCache::new(NonZeroUsize::new(new_size).unwrap()),
            allocated: HashSet::new(),
        };
        CACHE.allocator.insert(device_id, allocator);
    }
    CPU_LRU_CACHE_SIZE.store(new_size, Ordering::Relaxed);
}
