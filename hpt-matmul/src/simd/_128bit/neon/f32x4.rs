use std::arch::aarch64::*;
use crate::{ simd::_128bit::common::f32x4::f32x4, VecTrait };

impl VecTrait<f32> for f32x4 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { f32x4(vfmaq_f32(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn splat(val: f32) -> f32x4 {
        unsafe { f32x4(vdupq_n_f32(val)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        unsafe { f32x4(vfmaq_laneq_f32::<LANE>(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn partial_load(ptr: *const f32, num_elem: usize) -> Self {
        let mut result = Self::splat(f32::default());
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, &mut result.0 as *mut _ as *mut f32, num_elem);
            result
        }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut f32, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(&self.0 as *const _ as *const f32, ptr, num_elem);
        }
    }
}

impl std::ops::Add for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { f32x4(vaddq_f32(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { f32x4(vmulq_f32(self.0, rhs.0)) }
    }
}
