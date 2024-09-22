/// A struct contains a mutable simd vector
pub struct MutVec<'a, T> {
    vec: &'a mut T,
}

impl<T> MutVec<'_, T> {
    /// perform write unaligned operation
    pub fn write_unaligned(&mut self, value: T) {
        let ptr = self.vec as *mut T;
        unsafe {
            core::ptr::write_unaligned(ptr, value);
        }
    }
}
