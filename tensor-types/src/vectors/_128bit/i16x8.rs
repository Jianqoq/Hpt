use std::ops::{ Deref, DerefMut };

use crate::traits::{Init, VecCommon, VecTrait};

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct i16x8(pub(crate) std::simd::i16x8);

impl Deref for i16x8 {
    type Target = std::simd::i16x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i16x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i16> for i16x8 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i16]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const i16 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
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
    
    fn extract(self, idx: usize) -> i16 {
        self.as_array()[idx]
    }
}
impl VecCommon for i16x8 {
    const SIZE: usize = 8;
    
    type Base = i16;
}
impl Init<i16> for i16x8 {
    fn splat(val: i16) -> i16x8 {
        i16x8(std::simd::i16x8::splat(val))
    }
    unsafe fn from_ptr(ptr: *const i16) -> Self where Self: Sized {
        #[cfg(target_feature = "neon")]
        {
            unsafe { std::mem::transmute(std::arch::aarch64::vld1q_s16(ptr as *const _)) }
        }
        #[cfg(not(target_feature = "neon"))]
        {
            unsafe { std::mem::transmute(std::arch::x86_64::_mm_loadu_si128(ptr as *const _)) }
        }
    }
}
impl std::ops::Add for i16x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        i16x8(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i16x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        i16x8(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i16x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        i16x8(self.0 * rhs.0)
    }
}
impl std::ops::Div for i16x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        i16x8(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i16x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        i16x8(self.0 % rhs.0)
    }
}