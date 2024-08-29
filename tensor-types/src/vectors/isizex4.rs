use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct isizex4(pub(crate) std::simd::isizex4);
#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct isizex8(std::simd::i32x8);

impl Deref for isizex4 {
    #[cfg(target_pointer_width = "64")]
    type Target = std::simd::isizex4;
    #[cfg(target_pointer_width = "32")]
    type Target = std::simd::isizex8;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for isizex4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl VecTrait<isize> for isizex4 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        self.as_mut_array().copy_from_slice(unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const isize {
        self.as_array().as_ptr() as *const _
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut isize {
        self.as_mut_array().as_mut_ptr() as *mut _
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut isize {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> isize {
        #[cfg(target_pointer_width = "64")]
        let ret = self.as_array().iter().sum::<isize>();
        #[cfg(target_pointer_width = "32")]
        let ret = self.as_array().iter().sum::<isize>();
        ret as isize
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn as_array<N>(&self) -> &[isize; 8] {
        self.0.as_array()
    }
}

impl VecSize for isizex4 {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 4;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 8;
}

impl Init<isize> for isizex4 {
    fn splat(val: isize) -> isizex4 {
        #[cfg(target_pointer_width = "64")]
        let ret = isizex4(std::simd::isizex4::splat(val));
        #[cfg(target_pointer_width = "32")]
        let ret = isizex4(std::simd::isizex8::splat(val));
        ret
    }

    unsafe fn from_ptr(ptr: *const isize) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_loadu_si256(ptr as *const _)) }
    }
}

impl IntoVec<isizex4> for isizex4 {
    fn into_vec(self) -> isizex4 {
        self
    }
}
impl std::ops::Add for isizex4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        isizex4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for isizex4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        isizex4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for isizex4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        isizex4(self.0 * rhs.0)
    }
}
impl std::ops::Div for isizex4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        isizex4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for isizex4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        isizex4(self.0 % rhs.0)
    }
}