use std::panic::Location;

use hashbrown::{HashMap, HashSet};
use hpt_common::error::{base::TensorError, memory::MemoryError};
use lru::LruCache;

use crate::{
    ptr::SafePtr,
    storage::{CommonStorage, Storage},
};

fn allocate_mem(
    allocate_fn: impl Fn() -> *mut u8,
    deallocate_fn: &impl Fn(*mut u8, std::alloc::Layout),
    device_id: usize,
    layout: std::alloc::Layout,
    allocated: &mut HashSet<SafePtr>,
    cache: &mut LruCache<std::alloc::Layout, Vec<SafePtr>>,
) -> std::result::Result<*mut u8, TensorError> {
    let ptr = allocate_fn();
    if !ptr.is_null() {
        allocated.insert(SafePtr { ptr });
        Ok(ptr)
    } else {
        let needed_size = layout.size();
        let mut freed_size = 0;
        while freed_size < needed_size {
            if let Some((layout, ptrs)) = cache.pop_lru() {
                for safe_ptr in ptrs {
                    deallocate_fn(safe_ptr.ptr, layout);
                    freed_size += layout.size();
                    let ptr = allocate_fn();
                    if !ptr.is_null() {
                        allocated.insert(SafePtr { ptr });
                        return Ok(ptr);
                    }
                }
            } else {
                break;
            }
        }
        let ptr = allocate_fn();
        if !ptr.is_null() {
            allocated.insert(SafePtr { ptr });
            Ok(ptr)
        } else {
            Err(TensorError::Memory(MemoryError::AllocationFailed {
                device: "cpu".to_string(),
                id: device_id,
                size: layout.size() / 1024 / 1024,
                source: None,
                location: Location::caller(),
            }))
        }
    }
}

/// allocate memory based on layout provided, if the layout is not found in the cache, allocate, otherwise pop from the cache
///
/// this function internally checks if the cache is full, if it is full, it pops the least recently used layout and deallocates the memory
///
/// if the cache is not full, it inserts the allocated memory into the cache, and increments the reference count in the storage
///
/// # Safety
///
/// This function checks `null` ptr internally, any memory allocated through this method, downstream don't need to check for `null` ptr
#[track_caller]
pub(crate) fn allocate_helper(
    cache: &mut LruCache<std::alloc::Layout, Vec<SafePtr>>,
    allocated: &mut HashSet<SafePtr>,
    storage: &mut HashMap<usize, CommonStorage>,
    allocate_fn: impl Fn() -> *mut u8,
    zero_fn: impl Fn(*mut u8, std::alloc::Layout),
    deallocate_fn: impl Fn(*mut u8, std::alloc::Layout),
    layout: std::alloc::Layout,
    device_id: usize,
) -> std::result::Result<*mut u8, TensorError> {
    let ptr = if let Some(ptr) = cache.get_mut(&layout)
    /*check if we previously allocated same layout of memory */
    {
        // try pop the memory out, if it return None, there is no available cached memory, we need to allocate new memory
        if let Some(safe_ptr) = ptr.pop() {
            zero_fn(safe_ptr.ptr, layout);
            safe_ptr.ptr
        } else {
            allocate_mem(
                allocate_fn,
                &deallocate_fn,
                device_id,
                layout,
                allocated,
                cache,
            )?
        }
    } else {
        allocate_mem(
            allocate_fn,
            &deallocate_fn,
            device_id,
            layout,
            allocated,
            cache,
        )?
    };
    // check if the cache is full, if it is full, pop the least recently used layout and deallocate the memory
    if cache.cap().get() == cache.len() {
        if let Some((layout, ptrs)) = cache.pop_lru() {
            for safe_ptr in ptrs {
                deallocate_fn(safe_ptr.ptr, layout);
            }
        }
    }
    // increment the reference count in the storage of the ptr allocated
    if let Some(storage) = storage.get_mut(&device_id) {
        storage.increment_ref(SafePtr { ptr });
    } else {
        let mut new_storage = CommonStorage::new();
        new_storage.increment_ref(SafePtr { ptr });
        storage.insert(device_id, new_storage);
    }
    Ok(ptr)
}
