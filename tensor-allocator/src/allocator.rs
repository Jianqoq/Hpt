use std::{alloc::Layout, num::NonZeroUsize, sync::Mutex};

use hashbrown::HashSet;
use lru::LruCache;
use once_cell::sync::Lazy;

use crate::strorage::CPU_STORAGE;

/// just a wrapper around `*mut u8`, implementing `Send` and `Sync` trait to let the compiler know that it is safe to send and share across threads
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SafePtr {
    ptr: *mut u8,
}
unsafe impl Send for SafePtr {}
unsafe impl Sync for SafePtr {}

/// `lru` cache allocator
pub static CACHE: Lazy<Allocator> = Lazy::new(|| Allocator::new(100));

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
pub struct Allocator {
    allocator: Mutex<_Allocator>,
}

impl Allocator {
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
    pub fn allocate(&self, layout: Layout) -> anyhow::Result<*mut u8> {
        if let Ok(mut allocator) = self.allocator.lock() {
            allocator.allocate(layout)
        } else {
            anyhow::bail!("Failed to lock the allocator");
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
    pub fn deallocate(&self, ptr: *mut u8, layout: &Layout) {
        self.allocator.lock().unwrap().deallocate(ptr, layout);
    }

    /// if the ptr is found in the storage, increment the reference count, otherwise insert the ptr into the storage
    pub fn insert_ptr(&self, ptr: *mut u8) {
        self.allocator.lock().unwrap().insert_ptr(ptr);
    }

    /// clear the cache, deallocate all the memory allocated
    ///
    /// this is used when the program exits, it will be called automatically
    pub fn clear(&self) {
        match self.allocator.lock() {
            Ok(mut allocator) => {
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
            Err(err) => {
                println!("Failed to lock the allocator: {:?}", err);
            }
        }
    }
}

impl Allocator {
    pub fn new(capacity: usize) -> Self {
        Allocator {
            allocator: Mutex::new(_Allocator {
                cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
                allocated: HashSet::new(),
            }),
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
    fn allocate(&mut self, layout: Layout) -> anyhow::Result<*mut u8> {
        let ptr = if let Some(ptr) = self.cache.get_mut(&layout)
        /*check if we previously allocated same layout of memory */
        {
            // try pop the memory out, if it return None, there is no available cached memory, we need to allocate new memory
            if let Some(safe_ptr) = ptr.pop() {
                safe_ptr.ptr
            } else {
                let ptr = unsafe { std::alloc::alloc(layout) };
                if ptr.is_null() {
                    anyhow::bail!(
                        "Failed to allocate memory, for {} MB",
                        layout.size() / 1024 / 1024
                    );
                }
                self.allocated.insert(SafePtr { ptr });
                ptr
            }
        } else {
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                anyhow::bail!(
                    "Failed to allocate memory, for {} MB",
                    layout.size() / 1024 / 1024
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
            if let Some(cnt) = storage.get_mut(&SafePtr { ptr }) {
                *cnt = match cnt.checked_add(1) {
                    Some(cnt) => cnt,
                    None => anyhow::bail!("Reference count overflow"),
                };
            } else {
                storage.insert(SafePtr { ptr }, 1);
            }
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
            if let Some(cnt) = storage.get_mut(&SafePtr { ptr }) {
                *cnt = cnt.checked_sub(1).expect("Reference count underflow");
                if *cnt == 0 {
                    self.allocated.remove(&SafePtr { ptr });
                    storage.remove(&SafePtr { ptr });
                    if let Some(ptrs) = self.cache.get_mut(layout) {
                        ptrs.push(SafePtr { ptr });
                    } else {
                        self.cache.put(layout.clone(), vec![SafePtr { ptr }]);
                    }
                }
            } else {
                panic!("ptr {:p} not found in storage", ptr);
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
            if let Some(cnt) = storage.get_mut(&SafePtr { ptr }) {
                *cnt += 1;
            } else {
                storage.insert(SafePtr { ptr }, 1);
            }
        }
    }
}

/// # Clone Storage
///
/// increment the reference count of the ptr in the storage
pub fn clone_storage(ptr: *mut u8) {
    if let Ok(mut storage) = CPU_STORAGE.lock() {
        if let Some(cnt) = storage.get_mut(&SafePtr { ptr }) {
            *cnt += 1;
        } else {
            panic!("Pointer not found in CPU_STORAGE");
        }
    }
}
