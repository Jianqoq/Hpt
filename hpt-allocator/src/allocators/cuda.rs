use std::{
    alloc::Layout,
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    panic::Location,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use crate::{
    ptr::SafePtr,
    storage::{CommonStorage, Storage},
    traits::Allocator,
    CUDA_STORAGE,
};
use hpt_common::error::base::TensorError;
use lru::LruCache;
use once_cell::sync::Lazy;

/// `lru` cache allocator
pub(crate) static CUDA_CACHE: Lazy<Mutex<CudaAllocator>> =
    Lazy::new(|| Mutex::new(CudaAllocator::new()));

pub(crate) static CUDA_LRU_CACHE_SIZE: AtomicUsize = AtomicUsize::new(1);
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
pub struct CudaAllocator {
    allocator: HashMap<usize, (Arc<cudarc::driver::CudaDevice>, _Allocator)>,
}

impl Allocator for CudaAllocator {
    type Output = (*mut u8, Arc<cudarc::driver::CudaDevice>);
    type CpuAllocator = crate::allocators::cpu::CpuAllocator;
    #[cfg(feature = "cuda")]
    type CudaAllocator = CudaAllocator;
    fn allocate(&mut self, layout: Layout, device_id: usize) -> Result<Self::Output, TensorError> {
        if let Some((device, allocator)) = self.allocator.get_mut(&device_id) {
            Ok((
                allocator.allocate(layout, device_id, device.clone())?,
                device.clone(),
            ))
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(
                    NonZeroUsize::new(CUDA_LRU_CACHE_SIZE.load(Ordering::Relaxed)).unwrap(),
                ),
                allocated: HashSet::new(),
            };
            let device = cudarc::driver::CudaDevice::new(device_id).unwrap();
            let ptr = allocator.allocate(layout, device_id, device.clone())?;
            self.allocator
                .insert(device_id, (device.clone(), allocator));
            Ok((ptr, device))
        }
    }

    fn allocate_zeroed(
        &mut self,
        layout: Layout,
        device_id: usize,
    ) -> Result<Self::Output, TensorError> {
        if let Some((device, allocator)) = self.allocator.get_mut(&device_id) {
            Ok((
                allocator.allocate_zeroed(layout, device_id, device.clone())?,
                device.clone(),
            ))
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(
                    NonZeroUsize::new(CUDA_LRU_CACHE_SIZE.load(Ordering::Relaxed)).unwrap(),
                ),
                allocated: HashSet::new(),
            };
            let device = cudarc::driver::CudaDevice::new(device_id).unwrap();
            let ptr = allocator.allocate_zeroed(layout, device_id, device.clone())?;
            self.allocator
                .insert(device_id, (device.clone(), allocator));
            Ok((ptr, device))
        }
    }

    fn deallocate(&mut self, ptr: *mut u8, layout: &Layout, device_id: usize) {
        if let Some((_, allocator)) = self.allocator.get_mut(&device_id) {
            allocator.deallocate(ptr, layout, device_id);
        } else {
            panic!("Allocator for device {} not found", device_id);
        }
    }

    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        if let Some((_, allocator)) = self.allocator.get_mut(&device_id) {
            allocator.insert_ptr(ptr, device_id);
        } else {
            panic!("Allocator for device {} not found", device_id);
        }
    }

    fn clear(&mut self) {
        for (device, allocator) in self.allocator.values_mut() {
            for (layout, ptrs) in allocator.cache.iter_mut() {
                for ptr in ptrs.iter() {
                    unsafe { device.upgrade_device_ptr::<u8>(ptr.ptr as u64, layout.size()) };
                }
            }
            allocator.cache.clear();
            assert_eq!(allocator.allocated.len(), 0);
        }
    }

    fn new() -> Self {
        CudaAllocator {
            allocator: HashMap::new(),
        }
    }
}

#[derive(Clone)]
struct _Allocator {
    cache: LruCache<Layout, Vec<SafePtr>>,
    allocated: HashSet<SafePtr>,
}

impl _Allocator {
    #[track_caller]
    fn allocate(
        &mut self,
        layout: Layout,
        device_id: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<*mut u8, TensorError> {
        if let Ok(mut storage) = CUDA_STORAGE.lock() {
            let res = crate::utils::allocate::allocate_helper(
                &mut self.cache,
                &mut self.allocated,
                &mut storage,
                || {
                    let res = unsafe {
                        device
                            .alloc::<u8>(layout.size())
                            .map_err(
                                |e| hpt_common::error::device::DeviceError::CudaDriverError {
                                    message: format!(
                                        "Failed to allocate memory, for {} MB",
                                        layout.size() / 1024 / 1024
                                    ),
                                    source: Some(e),
                                    location: Location::caller(),
                                },
                            )
                            .expect("Failed to allocate memory")
                    };
                    res.leak() as *mut u8
                },
                |_, _| {},
                |ptr, layout| {
                    let slice =
                        unsafe { device.upgrade_device_ptr::<u8>(ptr as u64, layout.size()) };
                    drop(slice);
                },
                layout,
                device_id,
            );
            res
        } else {
            panic!("Failed to lock CPU_STORAGE");
        }
    }

    #[track_caller]
    fn allocate_zeroed(
        &mut self,
        layout: Layout,
        device_id: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> Result<*mut u8, TensorError> {
        if let Ok(mut storage) = CUDA_STORAGE.lock() {
            let res = crate::utils::allocate::allocate_helper(
                &mut self.cache,
                &mut self.allocated,
                &mut storage,
                || {
                    let res = device
                        .alloc_zeros::<u8>(layout.size())
                        .map_err(
                            |e| hpt_common::error::device::DeviceError::CudaDriverError {
                                message: format!(
                                    "Failed to allocate memory, for {} MB",
                                    layout.size() / 1024 / 1024
                                ),
                                source: Some(e),
                                location: Location::caller(),
                            },
                        )
                        .expect("Failed to allocate memory");
                    res.leak() as *mut u8
                },
                |ptr, layout| {
                    let mut slice =
                        unsafe { device.upgrade_device_ptr::<u8>(ptr as u64, layout.size()) };
                    device
                        .memset_zeros(&mut slice)
                        .expect("Failed to memset zeros");
                    slice.leak();
                },
                |ptr, layout| {
                    let slice =
                        unsafe { device.upgrade_device_ptr::<u8>(ptr as u64, layout.size()) };
                    drop(slice);
                },
                layout,
                device_id,
            );
            res
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
        if let Ok(mut storage) = CUDA_STORAGE.lock() {
            crate::utils::deallocate::deallocate_helper(
                &mut self.cache,
                &mut self.allocated,
                &mut storage,
                layout,
                ptr,
                device_id,
            );
        } else {
            panic!("Failed to lock CUDA_STORAGE");
        }
    }

    /// # Insert Pointer
    ///
    /// insert the ptr into the allocated set, and increment the reference count in the storage
    ///
    /// this function is used to insert the ptr into the allocated set, and increment the reference count in the storage
    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        self.allocated.insert(SafePtr { ptr });
        if let Ok(mut map) = CUDA_STORAGE.lock() {
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

/// resize the lru cache of the cuda allocator
///
/// when `new_size` >= `old_size`, cache size will increase and data won't be deallocated
///
/// when `new_size` < `old_size`, all the data in cache will be deallocated
pub fn resize_cuda_lru_cache(new_size: usize, device_id: usize) {
    if let Ok(mut cache) = CUDA_CACHE.lock() {
        if let Some((device, allocator)) = cache.allocator.get_mut(&device_id) {
            crate::utils::cache_resize::resize_lru_cache(
                &mut allocator.cache,
                |ptr, layout| {
                    let slice =
                        unsafe { device.upgrade_device_ptr::<u8>(ptr as u64, layout.size()) };
                    drop(slice);
                },
                new_size,
            );
        } else {
            panic!("device {} not found in cuda allocator", device_id);
        }
    } else {
        panic!("Failed to lock CUDA_CACHE");
    }
}
