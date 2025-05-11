
use std::arch::aarch64::*;

use crate::simd::_128bit::common::u16x8::u16x8;

impl u16x8 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_u16(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: u16) -> u16x8 {
        unsafe { u16x8(vdupq_n_u16(val)) }
    }
    #[inline(always)]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_laneq_u16::<LANE>(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const u16) -> Self {
        unsafe { u16x8(vld1q_u16(ptr)) }
    }
}

impl std::ops::Add for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(vaddq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(vmulq_u16(self.0, rhs.0)) }
    }
}