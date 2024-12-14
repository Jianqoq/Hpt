use std::sync::Mutex;

use hashbrown::HashMap;
use once_cell::sync::Lazy;

/// This is a global variable that stores the allocated ptrs and their reference count
pub static CPU_STORAGE: Lazy<Mutex<HashMap<crate::allocator::SafePtr, usize>>> =
    Lazy::new(|| HashMap::new().into());

/// This is a global variable that stores the allocated ptrs and their reference count
pub static CUDA_STORAGE: Lazy<Mutex<HashMap<usize, HashMap<crate::cuda_allocator::SafePtr, usize>>>> =
    Lazy::new(|| HashMap::new().into());
