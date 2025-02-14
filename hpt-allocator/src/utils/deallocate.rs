use hashbrown::{HashMap, HashSet};
use lru::LruCache;

use crate::{
    ptr::SafePtr,
    storage::{CommonStorage, Storage},
};

pub(crate) fn deallocate_helper(
    cache: &mut LruCache<std::alloc::Layout, Vec<SafePtr>>,
    allocated: &mut HashSet<SafePtr>,
    storage: &mut HashMap<usize, CommonStorage>,
    layout: &std::alloc::Layout,
    ptr: *mut u8,
    device_id: usize,
) {
    if let Some(storage) = storage.get_mut(&device_id) {
        if storage.decrement_ref(SafePtr { ptr }) {
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
