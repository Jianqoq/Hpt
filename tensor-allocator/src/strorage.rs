use std::sync::Mutex;

use hashbrown::HashMap;
use once_cell::sync::Lazy;

pub static mut CPU_STORAGE: Lazy<Mutex<HashMap<*mut u8, usize>>> = Lazy::new(||
    HashMap::new().into()
);
