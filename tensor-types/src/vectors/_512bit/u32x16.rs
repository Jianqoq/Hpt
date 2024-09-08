use std::ops::{ Deref, DerefMut };

use crate::vectors::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct u32x16(pub(crate) std::simd::u32x16);

impl Deref for u32x16 {
    type Target = std::simd::u32x16;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u32x16 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u32> for u32x16 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u32]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u32 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
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

    fn extract(self, idx: usize) -> u32 {
        self.as_array()[idx]
    }
}
impl VecSize for u32x16 {
    const SIZE: usize = 8;
}
impl Init<u32> for u32x16 {
    fn splat(val: u32) -> u32x16 {
        u32x16(std::simd::u32x16::splat(val))
    }

    unsafe fn from_ptr(ptr: *const u32) -> Self {
        unsafe {
            std::mem::transmute(std::simd::u32x16::from_slice(std::slice::from_raw_parts(ptr, 16)))
        }
    }
}
impl std::ops::Add for u32x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        u32x16(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u32x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        u32x16(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u32x16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        u32x16(self.0 * rhs.0)
    }
}
impl std::ops::Div for u32x16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        u32x16(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u32x16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        u32x16(self.0 % rhs.0)
    }
}
