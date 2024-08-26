use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct usizex4(std::simd::usizex4);
#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct usizex2(std::simd::usizex8);

impl Deref for usizex4 {
    #[cfg(target_pointer_width = "64")]
    type Target = std::simd::usizex4;
    #[cfg(target_pointer_width = "32")]
    type Target = std::simd::usizex8;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for usizex4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl VecTrait<usize> for usizex4 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[usize]) {
        self.as_mut_array().copy_from_slice(unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const usize {
        self.as_array().as_ptr() as *const _
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut usize {
        self.as_mut_array().as_mut_ptr() as *mut _
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut usize {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> usize {
        self.as_array().iter().sum::<usize>()
    }
}

impl VecSize for usizex4 {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 4;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 8;
}

impl Init<usize> for usizex4 {
    fn splat(val: usize) -> usizex4 {
        #[cfg(target_pointer_width = "64")]
        let ret = usizex4(std::simd::usizex4::splat(val));
        #[cfg(target_pointer_width = "32")]
        let ret = usizex8(std::simd::usizex8::splat(val));
        ret
    }

    unsafe fn from_ptr(ptr: *const usize) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}

impl IntoVec<usizex4> for usizex4 {
    fn into_vec(self) -> usizex4 {
        self
    }
}
