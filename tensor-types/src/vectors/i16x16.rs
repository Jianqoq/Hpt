use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct i16x16(pub(crate) std::simd::i16x16);

impl Deref for i16x16 {
    type Target = std::simd::i16x16;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i16x16 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i16> for i16x16 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i16]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const i16 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut i16 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut i16 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> i16 {
        self.as_array().iter().sum()
    }
}
impl VecSize for i16x16 {
    const SIZE: usize = 16;
}
impl Init<i16> for i16x16 {
    fn splat(val: i16) -> i16x16 {
        i16x16(std::simd::i16x16::splat(val))
    }

    unsafe fn from_ptr(ptr: *const i16) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl IntoVec<i16x16> for i16x16 {
    fn into_vec(self) -> i16x16 {
        self
    }
}
impl std::ops::Add for i16x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        i16x16(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i16x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        i16x16(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i16x16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        i16x16(self.0 * rhs.0)
    }
}
impl std::ops::Div for i16x16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        i16x16(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i16x16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        i16x16(self.0 % rhs.0)
    }
}