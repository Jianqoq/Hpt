pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

use crate::ptr::SafePtr;

pub trait Storage {
    fn increment_ref(&mut self, ptr: SafePtr);
    fn decrement_ref(&mut self, ptr: SafePtr) -> bool;
}
