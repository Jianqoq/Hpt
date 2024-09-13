use std::{ alloc::Layout, num::NonZeroUsize, sync::Mutex };

use hashbrown::HashSet;
use lru::LruCache;
use once_cell::sync::Lazy;

use crate::strorage::CPU_STORAGE;

pub static mut CACHE: Lazy<Allocator> = Lazy::new(|| Allocator::new(100));

pub struct Allocator {
    allocator: Mutex<_Allocator>,
}

impl Allocator {
    pub fn allocate(&self, layout: Layout) -> *mut u8 {
        self.allocator.lock().unwrap().allocate(layout)
    }

    pub fn deallocate(&self, ptr: *mut u8, layout: &Layout) {
        self.allocator.lock().unwrap().deallocate(ptr, layout);
    }

    pub fn insert_ptr(&self, ptr: *mut u8) {
        self.allocator.lock().unwrap().insert_ptr(ptr);
    }

    pub fn clear(&self) {
        let mut allocator = self.allocator.lock().unwrap();
        for (layout, ptrs) in allocator.cache.iter_mut() {
            for ptr in ptrs.iter() {
                unsafe {
                    std::alloc::dealloc(*ptr, layout.clone());
                }
            }
        }
        allocator.cache.clear();
        assert_eq!(allocator.allocated.len(), 0);
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
    cache: LruCache<Layout, Vec<*mut u8>>,
    allocated: HashSet<*mut u8>,
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
    fn allocate(&mut self, layout: Layout) -> *mut u8 {
        let ptr = if
            let Some(ptr) = self.cache.get_mut(
                &layout
            ) /*check if we previously allocated same layout of memory */
        {
            // try pop the memory out, if it return None, there is no available cached memory, we need to allocate new memory
            if let Some(ptr) = ptr.pop() {
                ptr
            } else {
                let ptr = unsafe { std::alloc::alloc(layout) };
                if ptr.is_null() {
                    panic!("Failed to allocate memory, for {} MB", layout.size() / 1024 / 1024);
                }
                self.allocated.insert(ptr);
                ptr
            }
        } else {
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                panic!("Failed to allocate memory, for {} MB", layout.size() / 1024 / 1024);
            }
            self.allocated.insert(ptr);
            ptr
        };
        // check if the cache is full, if it is full, pop the least recently used layout and deallocate the memory
        if self.cache.cap().get() == self.cache.len() {
            if let Some((layout, ptrs)) = self.cache.pop_lru() {
                for ptr in ptrs {
                    unsafe {
                        std::alloc::dealloc(ptr, layout);
                    }
                }
            }
        }
        // increment the reference count in the storage of the ptr allocated
        unsafe {
            if let Ok(mut storage) = CPU_STORAGE.lock() {
                if let Some(cnt) = storage.get_mut(&ptr) {
                    *cnt = cnt.checked_sub(1).expect("Reference count underflow");
                } else {
                    storage.insert(ptr, 1);
                }
            }
        }
        ptr
    }

    /// # Main Deallocation Function
    ///
    /// deallocate memory based on the ptr provided, if the ptr is found in the storage, decrement the reference count
    ///
    /// if the reference count is 0, remove the ptr from the storage, remove the ptr from the allocated set, and insert the ptr into the cache
    fn deallocate(&mut self, ptr: *mut u8, layout: &Layout) {
        unsafe {
            if let Ok(mut storage) = CPU_STORAGE.lock() {
                if let Some(cnt) = storage.get_mut(&ptr) {
                    *cnt = cnt.checked_sub(1).expect("Reference count underflow");
                    if *cnt == 0 {
                        self.allocated.remove(&ptr);
                        storage.remove(&ptr);
                        if let Some(ptrs) = self.cache.get_mut(layout) {
                            ptrs.push(ptr);
                        } else {
                            self.cache.put(layout.clone(), vec![ptr]);
                        }
                    }
                } else {
                    panic!("ptr {:p} not found in storage", ptr);
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
        self.allocated.insert(ptr);
        // println!("Inserting ptr {:p}", ptr);
        unsafe {
            if let Ok(mut storage) = CPU_STORAGE.lock() {
                if let Some(cnt) = storage.get_mut(&ptr) {
                    *cnt += 1;
                } else {
                    storage.insert(ptr, 1);
                }
            }
        }
    }
}

pub fn clone_storage(ptr: *mut u8) {
    unsafe {
        if let Ok(mut storage) = CPU_STORAGE.lock() {
            if let Some(cnt) = storage.get_mut(&ptr) {
                *cnt += 1;
            } else {
                panic!("Pointer not found in CPU_STORAGE");
            }
        }
    }
}
