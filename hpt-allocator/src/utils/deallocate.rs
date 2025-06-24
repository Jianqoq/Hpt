use std::collections::HashSet;

use dashmap::DashMap;
use lru::LruCache;

use crate::{
    ptr::SafePtr,
    storage::{CommonStorage, Storage},
};

pub(crate) fn deallocate_helper(
    cache: &mut LruCache<std::alloc::Layout, Vec<SafePtr>>,
    allocated: &mut HashSet<SafePtr>,
    storage: &DashMap<usize, CommonStorage>,
    layout: &std::alloc::Layout,
    ptr: *mut u8,
    should_drop: bool,
    device_id: usize,
) {
    if let Some(mut storage) = storage.get_mut(&device_id) {
        if storage.decrement_ref(SafePtr { ptr }) && should_drop {
            allocated.remove(&SafePtr { ptr });
            if let Some(ptrs) = cache.get_mut(layout) {
                ptrs.push(SafePtr { ptr });
            } else {
                cache.put(layout.clone(), vec![SafePtr { ptr }]);
            }
        }
    } else {
        panic!("device {} not found in storage", device_id);
    }
}
