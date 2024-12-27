use std::marker::PhantomData;

use crate::pointer::Pointer;

/// A struct contains a mutable simd vector
#[derive(Debug)]
pub struct MutVec<'a, T> {
    ptr: Pointer<T>,
    _phantom: PhantomData<&'a mut T>,
}

impl<'a, T> MutVec<'a, T> {
    /// create a new MutVec
    #[inline(always)]
    pub fn new(ptr: Pointer<T>) -> Self {
        Self {
            ptr,
            _phantom: PhantomData,
        }
    }

    /// perform write unaligned operation
    #[inline(always)]
    pub fn write_unaligned(&self, value: T) {
        unsafe {
            self.ptr.ptr.write_unaligned(value);
        }
    }

    #[inline(always)]
    /// perform read unaligned operation
    pub fn read_unaligned(&self) -> T {
        unsafe { self.ptr.ptr.read_unaligned() }
    }
}
