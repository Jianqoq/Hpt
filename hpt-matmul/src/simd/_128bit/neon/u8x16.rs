
use std::arch::aarch64::*;

use crate::simd::_128bit::common::u8x16::u8x16;

impl u8x16 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_u8(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: u8) -> u8x16 {
        unsafe { u8x16(vdupq_n_u8(val)) }
    }
    #[inline(always)]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const u8) -> Self {
        unsafe { u8x16(vld1q_u8(ptr)) }
    }
}

impl std::ops::Add for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { u8x16(vaddq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { u8x16(vmulq_u8(self.0, rhs.0)) }
    }
}