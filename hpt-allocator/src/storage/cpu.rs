use std::sync::Mutex;

use hashbrown::HashMap;
use once_cell::sync::Lazy;

use super::CommonStorage;

/// This is a global variable that stores the allocated ptrs and their reference count for CPU devices
pub static CPU_STORAGE: Lazy<Mutex<HashMap<usize, CommonStorage>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
