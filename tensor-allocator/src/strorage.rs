use std::sync::Mutex;

use hashbrown::HashMap;
use once_cell::sync::Lazy;

use crate::allocator::SafePtr;

/// This is a global variable that stores the allocated ptrs and their reference count
pub static CPU_STORAGE: Lazy<Mutex<HashMap<SafePtr, usize>>> =
    Lazy::new(|| HashMap::new().into());
