use num_complex::Complex64;

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct cplx64x2(pub(crate) [Complex64; 2]);

impl VecTrait<Complex64> for cplx64x2 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex64]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const Complex64 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut Complex64 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut Complex64 {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> Complex64 {
        self.0.iter().sum()
    }
}
impl VecSize for cplx64x2 {
    const SIZE: usize = 2;
}
impl Init<Complex64> for cplx64x2 {
    fn splat(val: Complex64) -> cplx64x2 {
        cplx64x2([val; 2])
    }

    unsafe fn from_ptr(ptr: *const Complex64) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_pd(ptr as *const _)) }
    }
}
impl IntoVec<cplx64x2> for cplx64x2 {
    fn into_vec(self) -> cplx64x2 {
        self
    }
}
