use std::sync::Mutex;

use hashbrown::HashMap;
use once_cell::sync::Lazy;

use super::{SafePtr, Storage};

/// This is a global variable that stores the allocated ptrs and their reference count
pub static CPU_STORAGE: Lazy<Mutex<CpuStorage>> = Lazy::new(|| Mutex::new(CpuStorage::new()));

pub struct CpuStorage {
    storage: HashMap<SafePtr, usize>,
}

impl CpuStorage {
    pub fn new() -> Self {
        CpuStorage {
            storage: HashMap::new(),
        }
    }
}

impl Storage for CpuStorage {
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
pub fn clone_storage(ptr: *mut u8) {
    if let Ok(mut storage) = CPU_STORAGE.lock() {
        storage.increment_ref(SafePtr { ptr });
    }
}
