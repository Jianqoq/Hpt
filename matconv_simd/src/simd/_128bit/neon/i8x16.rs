use std::arch::aarch64::*;

use crate::{ simd::_128bit::common::i8x16::i8x16, VecTrait };

impl VecTrait<i8> for i8x16 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i8x16(vmlaq_s8(self.0, a.0, b.0)) }
    }
    #[inline(always)]
    fn splat(val: i8) -> i8x16 {
        unsafe { i8x16(vdupq_n_s8(val)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    fn partial_load(ptr: *const i8, num_elem: usize) -> Self {
        let mut result = Self::splat(i8::default());
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, (&mut result.0) as *mut _ as *mut i8, num_elem);
            result
        }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut i8, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping((&self.0) as *const _ as *const i8, ptr, num_elem);
        }
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
