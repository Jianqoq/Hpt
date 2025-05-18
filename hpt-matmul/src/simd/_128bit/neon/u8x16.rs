
use std::arch::aarch64::*;

use crate::{simd::_128bit::common::u8x16::u8x16, VecTrait};

impl VecTrait<u8> for u8x16 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_u8(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn splat(val: u8) -> u8x16 {
        unsafe { u8x16(vdupq_n_u8(val)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    fn partial_load(ptr: *const u8, num_elem: usize) -> Self {
        let mut result = Self::splat(u8::default());
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, (&mut result.0) as *mut _ as *mut u8, num_elem);
            result
        }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut u8, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping((&self.0) as *const _ as *const u8, ptr, num_elem);
        }
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