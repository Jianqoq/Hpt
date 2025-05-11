
use std::arch::aarch64::*;

use crate::simd::_128bit::common::u64x2::u64x2;

impl u64x2 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [u64; 2] = std::mem::transmute(self.0);
            let arr2: [u64; 2] = std::mem::transmute(a.0);
            let arr3: [u64; 2] = std::mem::transmute(b.0);
            let mut arr4: [u64; 2] = [0; 2];
            for i in 0..2 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            return u64x2(vld1q_u64(arr4.as_ptr()));
        }
    }
    #[inline(always)]
    pub(crate) fn splat(val: u64) -> u64x2 {
        unsafe { u64x2(vdupq_n_u64(val)) }
    }
    #[inline(always)]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const u64) -> Self {
        unsafe { u64x2(vld1q_u64(ptr)) }
    }
}

impl std::ops::Add for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { u64x2(vaddq_u64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 2] = std::mem::transmute(self.0);
            let arr2: [u64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i].wrapping_mul(arr2[i]);
            }
            return u64x2(vld1q_u64(arr3.as_ptr()));
        }
    }
}