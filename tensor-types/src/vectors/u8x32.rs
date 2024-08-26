use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct u8x32([u8; 32]);

impl VecTrait<u8> for u8x32 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u8]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut u8 {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> u8 {
        self.0.iter().sum()
    }
}
impl VecSize for u8x32 {
    const SIZE: usize = 32;
}
impl Init<u8> for u8x32 {
    fn splat(val: u8) -> u8x32 {
        u8x32([val; 32])
    }

    unsafe fn from_ptr(ptr: *const u8) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl IntoVec<u8x32> for u8x32 {
    fn into_vec(self) -> u8x32 {
        self
    }
}
