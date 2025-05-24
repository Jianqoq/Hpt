
use std::arch::aarch64::*;

use crate::{simd::_128bit::common::i32x4::i32x4, VecTrait};

impl VecTrait<i32> for i32x4 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_s32(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn splat(val: i32) -> i32x4 {
        unsafe { i32x4(vdupq_n_s32(val)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        Self(unsafe { vmlaq_laneq_s32::<LANE>(b.0, self.0, a.0) })
    }
    #[inline(always)]
    fn partial_load(ptr: *const i32, num_elem: usize) -> Self {
        let mut result = Self::splat(i32::default());
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, (&mut result.0) as *mut _ as *mut i32, num_elem);
            result
        }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut i32, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping((&self.0) as *const _ as *const i32, ptr, num_elem);
        }
    }
}

impl std::ops::Add for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(vaddq_s32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(vmulq_s32(self.0, rhs.0)) }
    }
}

impl std::ops::BitAnd for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(vandq_s32(self.0, rhs.0)) }
    }
}

impl i32x4 {
    #[inline(always)]
    pub(crate) fn simd_ne(self, other: Self) -> i32x4 {
        unsafe {
            let eq = vceqq_s32(self.0, other.0);
            i32x4(veorq_s32(std::mem::transmute(eq), vdupq_n_s32(-1)))
        }
    }
    #[inline(always)]
    pub(crate) fn simd_gt(self, other: Self) -> i32x4 {
        unsafe {
            let cmp = vcgtq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
}
