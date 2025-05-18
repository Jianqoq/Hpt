
use std::arch::aarch64::*;

use crate::{simd::_128bit::common::i16x8::i16x8, VecTrait};

impl VecTrait<i16> for i16x8 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_s16(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn splat(val: i16) -> i16x8 {
        unsafe { i16x8(vdupq_n_s16(val)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        Self(unsafe { vmlaq_laneq_s16::<LANE>(b.0, self.0, a.0) })
    }
    #[inline(always)]
    fn partial_load(ptr: *const i16, num_elem: usize) -> Self {
        let mut result = Self::splat(i16::default());
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, (&mut result.0) as *mut _ as *mut i16, num_elem);
            result
        }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut i16, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping((&self.0) as *const _ as *const i16, ptr, num_elem);
        }
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