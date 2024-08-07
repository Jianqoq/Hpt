use std::sync::Mutex;

use hashbrown::HashMap;
use once_cell::sync::Lazy;
use wgpu::{ Buffer, Id };

pub static mut WGPU_STORAGE: Lazy<Mutex<HashMap<Id<Buffer>, usize>>> = Lazy::new(||
    HashMap::new().into()
);
pub static mut CPU_STORAGE: Lazy<Mutex<HashMap<*mut u8, usize>>> = Lazy::new(||
    HashMap::new().into()
);
