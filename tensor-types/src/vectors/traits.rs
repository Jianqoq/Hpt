use crate::dtype::TypeCommon;

/// common trait for all vector types
pub trait VecTrait<T> {
    /// peform self * a + b, fused multiply add
    fn _mul_add(self, a: Self, b: Self) -> Self;
    /// copy data from slice to self
    fn copy_from_slice(&mut self, slice: &[T]);
    /// convert self to a const pointer
    fn as_ptr(&self) -> *const T;
    /// convert self to a mutable pointer
    fn as_mut_ptr(&mut self) -> *mut T;
    /// convert self to a mutable pointer without check
    fn as_mut_ptr_uncheck(&self) -> *mut T;
    /// extract a value from vector
    fn extract(self, idx: usize) -> T;
    /// get the sum of all elements in vector
    fn sum(&self) -> T;
    /// write value to vector, this is unaligned write
    #[inline(always)]
    fn write_unaligned(&mut self, vec: T::Vec)
    where
        T: TypeCommon,
    {
        let ptr = self.as_mut_ptr() as *mut T::Vec;
        unsafe { ptr.write_unaligned(vec) }
    }
}

/// a trait for vector initialization
pub trait Init<T> {
    /// create a vector with all elements set to the val
    fn splat(val: T) -> Self;
    /// load data to vector from pointer
    ///
    /// # Safety
    ///
    /// This function is unsafe because it can cause undefined behavior if the pointer is invalid or the data len is less than the vector size
    unsafe fn from_ptr(ptr: *const T) -> Self
    where
        Self: Sized,
    {
        let ptr = ptr as *const Self;
        unsafe { ptr.read_unaligned() }
    }
}

/// a trait to get the vector size
pub trait VecCommon {
    /// get the number of lanes of the vector
    const SIZE: usize;
    /// the base type of the vector
    type Base: TypeCommon;
}

/// a trait to select value from two vectors
pub trait SimdSelect<T> {
    /// select value based on mask
    fn select(&self, true_val: T, false_val: T) -> T;
}

pub trait SimdCompare {
    type SimdMask;
    /// compare two vectors to check if is equal and return a mask
    fn simd_eq(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is not equal and return a mask
    fn simd_ne(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is less than and return a mask
    fn simd_lt(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is less than or equal and return a mask
    fn simd_le(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is greater than and return a mask
    fn simd_gt(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is greater than or equal and return a mask
    fn simd_ge(self, other: Self) -> Self::SimdMask;
}
