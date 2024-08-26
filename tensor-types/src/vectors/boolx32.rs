use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct boolx32([bool; 32]);

impl VecTrait<bool> for boolx32 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[bool]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const bool {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut bool {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut bool {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> bool {
        self.0
            .iter()
            .map(|&x| x as u8)
            .sum::<u8>() > 0
    }
}
impl VecSize for boolx32 {
    const SIZE: usize = 32;
}
impl Init<bool> for boolx32 {
    fn splat(val: bool) -> boolx32 {
        boolx32([val; 32])
    }

    unsafe fn from_ptr(ptr: *const bool) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl IntoVec<boolx32> for boolx32 {
    fn into_vec(self) -> boolx32 {
        self
    }
}