use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct u16x16(wide::u16x16);

impl Deref for u16x16 {
    type Target = wide::u16x16;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u16x16 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u16> for u16x16 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u16]) {
        self.as_array_mut().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u16 {
        self.as_array_ref().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u16 {
        self.as_array_mut().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut u16 {
        self.as_array_ref().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        self.as_array_ref().iter().sum()
    }
}
impl VecSize for u16x16 {
    const SIZE: usize = 16;
}
impl Init<u16> for u16x16 {
    fn splat(val: u16) -> u16x16 {
        u16x16(wide::u16x16::splat(val))
    }

    unsafe fn from_ptr(ptr: *const u16) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl IntoVec<u16x16> for u16x16 {
    fn into_vec(self) -> u16x16 {
        self
    }
}