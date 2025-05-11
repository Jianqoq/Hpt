

use std::arch::aarch64::*;

use crate::simd::_128bit::common::f64x2::f64x2;

impl f64x2 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            f64x2(vfmaq_f64(b.0, self.0, a.0))
        }
    }
    #[inline(always)]
    pub(crate) fn splat(val: f64) -> f64x2 {
        unsafe { f64x2(vdupq_n_f64(val)) }
    }
    #[inline(always)]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        unsafe { Self(vfmaq_laneq_f64::<LANE>(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const f64) -> Self {
        unsafe { f64x2(vld1q_f64(ptr)) }
    }
}

impl std::ops::Add for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { f64x2(vaddq_f64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for f64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { f64x2(vmulq_f64(self.0, rhs.0)) }
    }
}