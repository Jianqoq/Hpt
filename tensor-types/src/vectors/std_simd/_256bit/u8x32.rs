use std::ops::{ Deref, DerefMut, Index, IndexMut };

use crate::vectors::traits::{ Init, VecTrait };

/// a vector of 32 u8 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct u8x32(pub(crate) std::simd::u8x32);

impl Deref for u8x32 {
    type Target = std::simd::u8x32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u8x32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u8> for u8x32 {
    const SIZE: usize = 32;
    type Base = u8;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u8]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut u8 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> u8 {
        self.as_array().iter().sum()
    }
}

impl Init<u8> for u8x32 {
    fn splat(val: u8) -> u8x32 {
        u8x32(std::simd::u8x32::splat(val))
    }
}
impl Index<usize> for u8x32 {
    type Output = u8;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.as_array()[idx]
    }
}
impl IndexMut<usize> for u8x32 {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[idx]
    }
}
impl std::ops::Add for u8x32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        u8x32(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u8x32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        u8x32(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u8x32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        u8x32(self.0 * rhs.0)
    }
}
impl std::ops::Div for u8x32 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        u8x32(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u8x32 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        u8x32(self.0 % rhs.0)
    }
}
