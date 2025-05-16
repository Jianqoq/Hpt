use std::arch::x86_64::{_mm512_add_epi64, _mm512_fmadd_ps, _mm512_mullo_epi64, _mm512_set1_epi64};
use crate::simd::_512bit::common::f32x16::f32x16;
use crate::simd::_512bit::common::i64x8::i64x8;

impl i64x8 {
    #[inline(always)]
    pub(crate) fn splat(val: i64) -> i64x8 {
        unsafe {
            Self(_mm512_set1_epi64(val))
        }
    }
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let mul = _mm512_mullo_epi64(self.0, a.0);
            Self(_mm512_add_epi64(mul, b.0))
        }
    }
}

impl std::ops::Add for i64x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i64x8(_mm512_add_epi64(self.0, rhs.0)) }
    }
}