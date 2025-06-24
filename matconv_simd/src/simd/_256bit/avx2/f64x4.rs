
use std::arch::x86_64::*;
use crate::simd::_256bit::common::f64x4::f64x4;

impl f64x4 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(not(target_feature = "fma"))]
        unsafe {
            f64x4(_mm256_add_pd(_mm256_mul_pd(self.0, a.0), b.0))
        }
        #[cfg(target_feature = "fma")]
        unsafe {
            f64x4(_mm256_fmadd_pd(self.0, a.0, b.0))
        }
    }
    #[inline(always)]
    pub(crate) fn splat(val: f64) -> f64x4 {
        unsafe { f64x4(_mm256_set1_pd(val)) }
    }
}

impl std::ops::Add for f64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { f64x4(_mm256_add_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for f64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { f64x4(_mm256_mul_pd(self.0, rhs.0)) }
    }
}