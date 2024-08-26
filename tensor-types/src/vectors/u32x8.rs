use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct u32x8(wide::u32x8);

impl Deref for u32x8 {
    type Target = wide::u32x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u32x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u32> for u32x8 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u32]) {
        self.as_array_mut().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u32 {
        self.as_array_ref().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u32 {
        self.as_array_mut().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut u32 {
        self.as_array_ref().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> u32 {
        self.as_array_ref().iter().sum()
    }
}
impl VecSize for u32x8 {
    const SIZE: usize = 8;
}
impl Init<u32> for u32x8 {
    fn splat(val: u32) -> u32x8 {
        u32x8(wide::u32x8::splat(val))
    }

    unsafe fn from_ptr(ptr: *const u32) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl IntoVec<u32x8> for u32x8 {
    fn into_vec(self) -> u32x8 {
        self
    }
}