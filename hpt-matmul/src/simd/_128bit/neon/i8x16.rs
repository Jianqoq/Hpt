
use std::arch::aarch64::*;

use crate::simd::_128bit::common::i8x16::i8x16;

impl i8x16 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i8x16(vmlaq_s8(self.0, a.0, b.0)) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: i8) -> i8x16 {
        unsafe { i8x16(vdupq_n_s8(val)) }
    }
    #[inline(always)]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const i8) -> Self {
        unsafe { i8x16(vld1q_s8(ptr)) }
    }
}

impl std::ops::Add for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(vaddq_s8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(vmulq_s8(self.0, rhs.0)) }
    }
}