/// A struct contains a mutable simd vector
#[derive(Debug)]
pub struct MutVec<'a, T> {
    vec: &'a mut T,
}

impl<T> MutVec<'_, T> {
    /// perform write unaligned operation
    #[inline(always)]
    pub fn write_unaligned(&self, value: T) {
        let ptr = self.vec as *const _ as *mut T;
        unsafe {
            ptr.write_unaligned(value);
        }
    }

    #[inline(always)]
    /// perform read unaligned operation
    pub fn read_unaligned(&self) -> T {
        let ptr = self.vec as *const T;
        unsafe { ptr.read_unaligned() }
    }

    /// get the inner value
    pub fn inner(&self) -> &T {
        self.vec
    }

    /// get the mutable inner value
    pub fn inner_mut(&mut self) -> &mut T {
        self.vec
    }

    /// get the pointer of the inner value
    pub fn ptr(&self) -> *const T {
        self.vec as *const T
    }

    /// get the mutable pointer of the inner value
    pub fn ptr_mut(&mut self) -> *mut T {
        self.vec as *mut T
    }
}
