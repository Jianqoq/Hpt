use std::{
    alloc::Layout,
    num::NonZeroUsize,
    panic::Location,
    sync::{Arc, Mutex},
};

use crate::{
    ptr::SafePtr,
    storage::{CommonStorage, Storage},
    CUDA_STORAGE,
};
use cudarc::driver::DeviceRepr;
use hashbrown::{HashMap, HashSet};
use hpt_common::error::base::TensorError;
use lru::LruCache;
use once_cell::sync::Lazy;

/// `lru` cache allocator
pub static CUDA_CACHE: Lazy<Mutex<CudaAllocator>> = Lazy::new(|| Mutex::new(CudaAllocator::new()));

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
pub struct CudaAllocator {
    allocator: HashMap<usize, (Arc<cudarc::driver::CudaDevice>, _Allocator)>,
}

impl CudaAllocator {
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
    pub fn allocate(
        &mut self,
        layout: Layout,
        device_id: usize,
    ) -> std::result::Result<(*mut u8, Arc<cudarc::driver::CudaDevice>), TensorError> {
        if let Some((device, allocator)) = self.allocator.get_mut(&device_id) {
            Ok((
                allocator.allocate(layout, device_id, device.clone())?,
                device.clone(),
            ))
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(NonZeroUsize::new(10).unwrap()),
                allocated: HashSet::new(),
            };
            let device = cudarc::driver::CudaDevice::new(device_id).unwrap();
            let ptr = allocator.allocate(layout, device_id, device.clone())?;
            self.allocator
                .insert(device_id, (device.clone(), allocator));
            Ok((ptr, device))
        }
    }

    pub fn memset_zeros(&mut self, ptr: *mut u8, layout: &Layout, device_id: usize) {
        if let Some((device, _)) = self.allocator.get_mut(&device_id) {
            let mut slice = unsafe { device.upgrade_device_ptr::<u8>(ptr as u64, layout.size()) };
            device.memset_zeros(&mut slice).unwrap();
            slice.leak();
        } else {
            panic!("Allocator for device {} not found", device_id);
        }
    }

    /// deallocate memory by using lru cache strategy
    ///
    /// # Logic
    ///
    /// 1. check if the ptr is found in the storage
    ///
    /// 2. if the ptr is found in the storage, decrement the reference count
    ///
    /// 3. if the reference count is 0, remove the ptr from the storage, remove the ptr from the allocated set, and insert the ptr into the cache
    pub fn deallocate(&mut self, ptr: *mut u8, layout: &Layout, device_id: usize) {
        if let Some((_, allocator)) = self.allocator.get_mut(&device_id) {
            allocator.deallocate(ptr, layout, device_id);
        } else {
            panic!("Allocator for device {} not found", device_id);
        }
    }

    /// if the ptr is found in the storage, increment the reference count, otherwise insert the ptr into the storage
    pub fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        if let Some((_, allocator)) = self.allocator.get_mut(&device_id) {
            allocator.insert_ptr(ptr, device_id);
        } else {
            panic!("Allocator for device {} not found", device_id);
        }
    }

    /// clear the cache, deallocate all the memory allocated
    ///
    /// this is used when the program exits, it will be called automatically
    pub fn clear(&mut self) {
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

    /// get the device by device_id
    pub fn host_to_device<T: DeviceRepr>(
        &mut self,
        value: &[T],
        device_id: usize,
    ) -> anyhow::Result<(*mut u8, Arc<cudarc::driver::CudaDevice>)> {
        if let Some((device, allocator)) = self.allocator.get_mut(&device_id) {
            Ok((
                allocator.htod(value, device_id, device.clone())?,
                device.clone(),
            ))
        } else {
            let mut allocator = _Allocator {
                cache: LruCache::new(NonZeroUsize::new(10).unwrap()),
                allocated: HashSet::new(),
            };
            let device = cudarc::driver::CudaDevice::new(device_id).unwrap();
            let ptr = allocator.htod(value, device_id, device.clone())?;
            self.allocator
                .insert(device_id, (device.clone(), allocator));
            Ok((ptr, device))
        }
    }
}

impl CudaAllocator {
    pub fn new() -> Self {
        CudaAllocator {
            allocator: HashMap::new(),
        }
    }
}

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
    ) -> std::result::Result<*mut u8, TensorError> {
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

    fn htod<T: DeviceRepr>(
        &mut self,
        value: &[T],
        device_id: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> anyhow::Result<*mut u8> {
        let layout = Layout::from_size_align(value.len() * size_of::<T>(), 32).unwrap();
        let slice = device.htod_sync_copy(value)?;
        let ptr = slice.leak() as *mut u8;
        if ptr.is_null() {
            anyhow::bail!(
                "Failed to allocate memory, for {} MB",
                layout.size() / 1024 / 1024
            );
        }
        self.allocated.insert(SafePtr { ptr });
        if let Ok(mut storage) = CUDA_STORAGE.lock() {
            if let Some(storage) = storage.get_mut(&device_id) {
                if let Some(cnt) = storage.storage.get_mut(&SafePtr { ptr }) {
                    *cnt = match cnt.checked_add(1) {
                        Some(cnt) => cnt,
                        None => anyhow::bail!("Reference count overflow"),
                    };
                } else {
                    storage.storage.insert(SafePtr { ptr }, 1);
                }
            }
        }
        Ok(ptr)
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
