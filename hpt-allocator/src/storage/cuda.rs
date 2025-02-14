use std::sync::Mutex;

use hashbrown::HashMap;
use once_cell::sync::Lazy;

use crate::storage::CommonStorage;

/// This is a global variable that stores the allocated ptrs and their reference count for CUDA devices
pub static CUDA_STORAGE: Lazy<Mutex<HashMap<usize, CommonStorage>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
