use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct i8x32(pub(crate) std::simd::i8x32);

impl Deref for i8x32 {
    type Target = std::simd::i8x32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i8x32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i8> for i8x32 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i8]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const i8 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut i8 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut i8 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        self.as_array().iter().sum()
    }
}
impl VecSize for i8x32 {
    const SIZE: usize = 32;
}
impl Init<i8> for i8x32 {
    fn splat(val: i8) -> i8x32 {
        i8x32(std::simd::i8x32::splat(val))
    }

    unsafe fn from_ptr(ptr: *const i8) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl IntoVec<i8x32> for i8x32 {
    fn into_vec(self) -> i8x32 {
        self
    }
}