use crate::dtype::TypeCommon;

pub trait VecTrait<T> {
    fn _mul_add(self, a: Self, b: Self) -> Self;
    fn copy_from_slice(&mut self, slice: &[T]);
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
    fn as_mut_ptr_uncheck(&self) -> *mut T;
    fn extract(self, idx: usize) -> T;
    fn sum(&self) -> T;
    fn write_unaligned(&mut self, vec: Self) where Self: Sized {
        let ptr = self.as_mut_ptr() as *mut Self;
        unsafe { ptr.write_unaligned(vec) }
    }
}
pub trait Init<T> {
    fn splat(val: T) -> Self;
    unsafe fn from_ptr(ptr: *const T) -> Self where Self: Sized {
        let mut tmp = core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr().cast(), 1);
            tmp.assume_init()
        }
    }
}
pub trait VecCommon {
    const SIZE: usize;
    type Base: TypeCommon;
}

pub trait SimdSelect<T> {
    fn select(&self, true_val: T, false_val: T) -> T;
}
