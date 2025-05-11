
use std::arch::aarch64::*;

use crate::simd::_128bit::common::i64x2::i64x2;

impl i64x2 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(a.0);
            let arr3: [i64; 2] = std::mem::transmute(b.0);
            let mut arr4: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            return i64x2(vld1q_s64(arr4.as_ptr()));
        }
    }
    #[inline(always)]
    pub(crate) fn splat(val: i64) -> i64x2 {
        unsafe { i64x2(vdupq_n_s64(val)) }
    }
    #[inline(always)]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const i64) -> Self {
        unsafe { i64x2(vld1q_s64(ptr)) }
    }
}

impl std::ops::Add for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(vaddq_s64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i].wrapping_mul(arr2[i]);
            }
            return i64x2(vld1q_s64(arr3.as_ptr()));
        }
    }
}