use std::arch::x86_64::*;
use crate::simd::_256bit::common::f32x8::f32x8;

impl f32x8 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(not(target_feature = "fma"))]
        unsafe {
            f32x8(_mm256_add_ps(_mm256_mul_ps(self.0, a.0), b.0))
        }
        #[cfg(target_feature = "fma")]
        unsafe {
            f32x8(_mm256_fmadd_ps(self.0, a.0, b.0))
        }
    }
    #[inline(always)]
    pub(crate) fn splat(val: f32) -> f32x8 {
        unsafe { f32x8(_mm256_set1_ps(val)) }
    }
}

impl std::ops::Add for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_add_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { f32x8(_mm256_mul_ps(self.0, rhs.0)) }
    }
}