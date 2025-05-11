
use std::arch::aarch64::*;

use crate::simd::_128bit::common::i16x8::i16x8;

impl i16x8 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_s16(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: i16) -> i16x8 {
        unsafe { i16x8(vdupq_n_s16(val)) }
    }
    #[inline(always)]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        Self(unsafe { vmlaq_laneq_s16::<LANE>(b.0, self.0, a.0) })
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const i16) -> Self {
        unsafe { i16x8(vld1q_s16(ptr)) }
    }
}

impl std::ops::Add for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(vaddq_s16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(vmulq_s16(self.0, rhs.0)) }
    }
}