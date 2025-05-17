use crate::{
    VecTrait,
    simd::_512bit::common::{f32x16::f32x16, mask::U16MASK},
};
use std::arch::x86_64::*;

impl VecTrait<f32> for f32x16 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(not(target_feature = "fma"))]
        unsafe {
            f32x16(_mm512_add_ps(_mm512_mul_ps(self.0, a.0), b.0))
        }
        #[cfg(target_feature = "fma")]
        unsafe {
            f32x16(_mm512_fmadd_ps(self.0, a.0, b.0))
        }
    }
    #[inline(always)]
    fn splat(val: f32) -> Self {
        unsafe { f32x16(_mm512_set1_ps(val)) }
    }
    #[inline(always)]
    fn partial_load(ptr: *const f32, num_elem: usize) -> Self {
        unsafe { f32x16(_mm512_maskz_load_ps(U16MASK[num_elem], ptr)) }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut f32, num_elem: usize) {
        unsafe { _mm512_mask_store_ps(ptr, U16MASK[num_elem], self.0) }
    }
}

impl std::ops::Add for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { f32x16(_mm512_add_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for f32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { f32x16(_mm512_mul_ps(self.0, rhs.0)) }
    }
}
