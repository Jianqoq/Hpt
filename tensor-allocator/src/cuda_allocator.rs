use std::{
    alloc::Layout,
    num::NonZeroUsize,
    panic::Location,
    sync::{Arc, Mutex},
};

use cudarc::driver::DeviceRepr;
use hashbrown::{HashMap, HashSet};
use lru::LruCache;
use once_cell::sync::Lazy;
use tensor_common::err_handler::ErrHandler;

use crate::CUDA_STORAGE;

/// just a wrapper around `*mut u8`, implementing `Send` and `Sync` trait to let the compiler know that it is safe to send and share across threads
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SafePtr {
    ptr: *mut u8,
}
unsafe impl Send for SafePtr {}
unsafe impl Sync for SafePtr {}

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
    ) -> std::result::Result<(*mut u8, Arc<cudarc::driver::CudaDevice>), ErrHandler> {
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
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn allocate(
        &mut self,
        layout: Layout,
        device_id: usize,
        device: Arc<cudarc::driver::CudaDevice>,
    ) -> std::result::Result<*mut u8, ErrHandler> {
        let ptr = if let Some(ptr) = self.cache.get_mut(&layout)
        /*check if we previously allocated same layout of memory */
        {
            // try pop the memory out, if it return None, there is no available cached memory, we need to allocate new memory
            if let Some(safe_ptr) = ptr.pop() {
                safe_ptr.ptr
            } else {
                let res = unsafe {
                    device.alloc::<u8>(layout.size()).map_err(|e| {
                        ErrHandler::CudaRcMemAllocFailed(
                            layout.size() / 1024 / 1024,
                            Location::caller(),
                            e,
                        )
                    })?
                };
                let ptr = res.leak() as *mut u8;
                if ptr.is_null() {
                    return Err(ErrHandler::MemAllocFailed(
                        "cpu",
                        layout.size() / 1024 / 1024,
                        Location::caller(),
                    ));
                }
                self.allocated.insert(SafePtr { ptr });
                ptr
            }
        } else {
            let res = unsafe {
                device.alloc::<u8>(layout.size()).map_err(|e| {
                    ErrHandler::CudaRcMemAllocFailed(
                        layout.size() / 1024 / 1024,
                        Location::caller(),
                        e,
                    )
                })?
            };
            let ptr = res.leak() as *mut u8;
            if ptr.is_null() {
                return Err(ErrHandler::MemAllocFailed(
                    "cpu",
                    layout.size() / 1024 / 1024,
                    Location::caller(),
                ));
            }
            self.allocated.insert(SafePtr { ptr });
            ptr
        };
        // check if the cache is full, if it is full, pop the least recently used layout and deallocate the memory
        if self.cache.cap().get() == self.cache.len() {
            if let Some((layout, ptrs)) = self.cache.pop_lru() {
                for safe_ptr in ptrs {
                    let slice = unsafe {
                        device.upgrade_device_ptr::<u8>(safe_ptr.ptr as u64, layout.size())
                    };
                    drop(slice);
                }
            }
        }
        // increment the reference count in the storage of the ptr allocated
        if let Ok(mut storage) = CUDA_STORAGE.lock() {
            if let Some(device) = storage.get_mut(&device_id) {
                if let Some(cnt) = device.get_mut(&SafePtr { ptr }) {
                    *cnt = match cnt.checked_add(1) {
                        Some(cnt) => cnt,
                        None => {
                            return Err(ErrHandler::ReferenceCountOverflow(
                                "cpu",
                                Location::caller(),
                            ))
                        }
                    };
                } else {
                    device.insert(SafePtr { ptr }, 1);
                }
            } else {
                let mut device = HashMap::new();
                device.insert(SafePtr { ptr }, 1);
                storage.insert(device_id, device);
            }
        }
        Ok(ptr)
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
            if let Some(device) = storage.get_mut(&device_id) {
                if let Some(cnt) = device.get_mut(&SafePtr { ptr }) {
                    *cnt = match cnt.checked_add(1) {
                        Some(cnt) => cnt,
                        None => anyhow::bail!("Reference count overflow"),
                    };
                } else {
                    device.insert(SafePtr { ptr }, 1);
                }
            } else {
                let mut device = HashMap::new();
                device.insert(SafePtr { ptr }, 1);
                storage.insert(device_id, device);
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
            if let Some(device) = storage.get_mut(&device_id) {
                if let Some(cnt) = device.get_mut(&SafePtr { ptr }) {
                    *cnt = cnt.checked_sub(1).expect("Reference count underflow");
                    if *cnt == 0 {
                        self.allocated.remove(&SafePtr { ptr });
                        device.remove(&SafePtr { ptr });
                        if let Some(ptrs) = self.cache.get_mut(layout) {
                            ptrs.push(SafePtr { ptr });
                        } else {
                            self.cache.put(layout.clone(), vec![SafePtr { ptr }]);
                        }
                    }
                } else {
                    panic!("ptr {:p} not found in storage", ptr);
                }
            } else {
                panic!("device {} not found in storage", device_id);
            }
        }
    }

    /// # Insert Pointer
    ///
    /// insert the ptr into the allocated set, and increment the reference count in the storage
    ///
    /// this function is used to insert the ptr into the allocated set, and increment the reference count in the storage
    fn insert_ptr(&mut self, ptr: *mut u8, device_id: usize) {
        self.allocated.insert(SafePtr { ptr });
        if let Ok(mut storage) = CUDA_STORAGE.lock() {
            if let Some(device) = storage.get_mut(&device_id) {
                if let Some(cnt) = device.get_mut(&SafePtr { ptr }) {
                    *cnt += 1;
                } else {
                    device.insert(SafePtr { ptr }, 1);
                }
            }
        }
    }
}

/// # Clone Storage
///
/// increment the reference count of the ptr in the storage
pub fn clone_storage(ptr: *mut u8, device_id: usize) {
    if let Ok(mut storage) = CUDA_STORAGE.lock() {
        if let Some(device) = storage.get_mut(&device_id) {
            if let Some(cnt) = device.get_mut(&SafePtr { ptr }) {
                *cnt += 1;
            } else {
                panic!("Pointer not found in CPU_STORAGE");
            }
        }
    }
}
