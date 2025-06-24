
use std::arch::aarch64::*;

use crate::{simd::_128bit::common::u16x8::u16x8, VecTrait};

impl VecTrait<u16> for u16x8 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_u16(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn splat(val: u16) -> u16x8 {
        unsafe { u16x8(vdupq_n_u16(val)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_laneq_u16::<LANE>(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn partial_load(ptr: *const u16, num_elem: usize) -> Self {
        let mut result = Self::splat(u16::default());
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, (&mut result.0) as *mut _ as *mut u16, num_elem);
            result
        }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut u16, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping((&self.0) as *const _ as *const u16, ptr, num_elem);
        }
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