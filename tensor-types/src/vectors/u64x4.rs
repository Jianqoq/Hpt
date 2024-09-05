use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct u64x4(pub(crate) std::simd::u64x4);

impl Deref for u64x4 {
    type Target = std::simd::u64x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u64x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u64> for u64x4 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u64]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u64 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u64 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut u64 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> u64 {
        self.as_array().iter().sum()
    }
}
impl VecSize for u64x4 {
    const SIZE: usize = 4;
}
impl Init<u64> for u64x4 {
    fn splat(val: u64) -> u64x4 {
        u64x4(std::simd::u64x4::splat(val))
    }

    unsafe fn from_ptr(ptr: *const u64) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_loadu_si256(ptr as *const _)) }
    }
}
impl IntoVec<u64x4> for u64x4 {
    fn into_vec(self) -> u64x4 {
        self
    }
}
impl std::ops::Add for u64x4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        u64x4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u64x4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        u64x4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u64x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        u64x4(self.0 * rhs.0)
    }
}
impl std::ops::Div for u64x4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        u64x4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u64x4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        u64x4(self.0 % rhs.0)
    }
}