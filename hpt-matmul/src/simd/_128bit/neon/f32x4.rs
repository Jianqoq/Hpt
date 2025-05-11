use std::arch::aarch64::*;
use crate::simd::_128bit::common::f32x4::f32x4;

impl f32x4 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { f32x4(vfmaq_f32(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: f32) -> f32x4 {
        unsafe { f32x4(vdupq_n_f32(val)) }
    }
    #[inline(always)]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        unsafe { f32x4(vfmaq_laneq_f32::<LANE>(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const f32) -> Self {
        unsafe { f32x4(vld1q_f32(ptr)) }
    }
}

impl std::ops::Add for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { f32x4(vaddq_f32(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { f32x4(vmulq_f32(self.0, rhs.0)) }
    }
}
