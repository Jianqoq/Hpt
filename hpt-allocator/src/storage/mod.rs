pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

use hashbrown::HashMap;

use crate::ptr::SafePtr;

pub trait Storage {
    fn increment_ref(&mut self, ptr: SafePtr);
    fn decrement_ref(&mut self, ptr: SafePtr) -> bool;
}

#[derive(Debug)]
pub struct CommonStorage {
    pub(crate) storage: HashMap<SafePtr, usize>,
}

impl CommonStorage {
    pub fn new() -> Self {
        CommonStorage {
            storage: HashMap::new(),
        }
    }
}

impl Storage for CommonStorage {
    fn increment_ref(&mut self, ptr: SafePtr) {
        if let Some(cnt) = self.storage.get_mut(&ptr) {
            *cnt = match cnt.checked_add(1) {
                Some(cnt) => cnt,
                None => {
                    panic!(
                        "Reference count overflow for ptr {:p} in cpu storage",
                        ptr.ptr
                    );
                }
            };
        } else {
            self.storage.insert(ptr, 1);
        }
    }

    fn decrement_ref(&mut self, ptr: SafePtr) -> bool {
        if let Some(cnt) = self.storage.get_mut(&ptr) {
            *cnt = cnt.checked_sub(1).expect("Reference count underflow");
            if *cnt == 0 {
                self.storage.remove(&ptr);
                true
            } else {
                false
            }
        } else {
            panic!("ptr {:p} not found in cpu storage", ptr.ptr);
        }
    }
}

/// # Clone Storage
///
/// increment the reference count of the ptr in the storage
pub fn clone_storage(ptr: *mut u8, device_id: usize, map: &mut HashMap<usize, CommonStorage>) {
    if let Some(storage) = map.get_mut(&device_id) {
        storage.increment_ref(SafePtr { ptr });
    } else {
        panic!("device {} not found in storage of cpu", device_id);
    }
}
