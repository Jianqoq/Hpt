use std::collections::HashMap;

use crate::{ptr::SafePtr, storage::CommonStorage};

/// # Forget Helper
///
/// forget the ptr from the storage, remove the ptr from the allocated set, and insert the ptr into the cache
pub(crate) fn forget_helper(
    storage: &mut HashMap<usize, CommonStorage>,
    ptr: *mut u8,
    device_id: usize,
) {
    if let Some(storage) = storage.get_mut(&device_id) {
        if let Some(cnt) = storage.storage.get_mut(&SafePtr { ptr }) {
            let cnt = cnt.checked_sub(1).expect("Reference count underflow");
            if cnt == 0 {
                storage.storage.remove(&SafePtr { ptr });
            } else {
                panic!(
                    "can't forget ptr {:p} because the reference count is not 0, cnt: {}",
                    ptr, cnt
                );
            }
        } else {
            panic!("ptr {:p} not found in cpu storage", ptr);
        }
    } else {
        panic!("device {} not found in storage", device_id);
    }
}
