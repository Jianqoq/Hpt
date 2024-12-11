use std::ops::{ Deref, DerefMut, Index, IndexMut };

use crate::vectors::traits::{ Init, VecTrait };

/// a vector of 8 u32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct u32x8(pub(crate) std::simd::u32x8);

impl Deref for u32x8 {
    type Target = std::simd::u32x8;
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
    const SIZE: usize = 8;

    type Base = u32;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u32]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u32 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u32 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut u32 {
        self.as_array().as_ptr() as *mut _
    }

    #[inline(always)]
    fn sum(&self) -> u32 {
        self.as_array().iter().sum()
    }
}

impl Init<u32> for u32x8 {
    fn splat(val: u32) -> u32x8 {
        u32x8(std::simd::u32x8::splat(val))
    }
}
impl Index<usize> for u32x8 {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}
impl IndexMut<usize> for u32x8 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}
impl std::ops::Add for u32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        u32x8(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        u32x8(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u32x8 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        u32x8(self.0 * rhs.0)
    }
}
impl std::ops::Div for u32x8 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        u32x8(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u32x8 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        u32x8(self.0 % rhs.0)
    }
}