use crate::dtype::TypeCommon;

pub trait VecTrait<T> {
    fn _mul_add(self, a: Self, b: Self) -> Self;
    fn copy_from_slice(&mut self, slice: &[T]);
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
    fn as_mut_ptr_uncheck(&self) -> *mut T;
    fn extract(self, idx: usize) -> T;
    fn sum(&self) -> T;
    #[inline(always)]
    fn write_unaligned(&mut self, vec: T::Vec) where T: TypeCommon {
        let ptr = self.as_mut_ptr() as *mut T::Vec;
        unsafe { ptr.write_unaligned(vec) }
    }
}
pub trait Init<T> {
    fn splat(val: T) -> Self;
    unsafe fn from_ptr(ptr: *const T) -> Self where Self: Sized {
        let ptr = ptr as *const Self;
        unsafe { ptr.read_unaligned() }
    }
}
pub trait VecCommon {
    const SIZE: usize;
    type Base: TypeCommon;
}

pub trait SimdSelect<T> {
    fn select(&self, true_val: T, false_val: T) -> T;
}
