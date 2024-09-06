
pub trait VecTrait<T> {
    fn _mul_add(self, a: Self, b: Self) -> Self;
    fn copy_from_slice(&mut self, slice: &[T]);
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
    fn as_mut_ptr_uncheck(&self) -> *mut T;
    fn extract(self, idx: usize) -> T;
    fn sum(&self) -> T;
}
pub trait Init<T> {
    fn splat(val: T) -> Self;
    unsafe fn from_ptr(ptr: *const T) -> Self;
}
pub trait VecSize {
    const SIZE: usize;
}

pub trait SimdSelect<T> {
    fn select(&self, true_val: T, false_val: T) -> T;
}