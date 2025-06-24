
use std::arch::x86_64::*;
use crate::simd::_256bit::common::i32x8::i32x8;

impl i32x8 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i32x8(_mm256_add_epi32(self.0, _mm256_mullo_epi32(a.0, b.0))) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: i32) -> i32x8 {
        unsafe { i32x8(_mm256_set1_epi32(val)) }
    }
}

impl i32x8 {
    #[inline(always)]
    pub(crate) fn simd_ne(self, other: Self) -> i32x8 {
        unsafe {
            let eq = _mm256_cmpeq_epi32(self.0, other.0);
            i32x8(_mm256_xor_si256(eq, _mm256_set1_epi32(-1)))
        }
    }
    #[inline(always)]
    pub(crate) fn simd_lt(self, other: Self) -> i32x8 {
        unsafe { i32x8(_mm256_cmpgt_epi32(other.0, self.0)) }
    }
    #[inline(always)]
    pub(crate) fn simd_le(self, other: Self) -> i32x8 {
        unsafe {
            let lt = _mm256_cmpgt_epi32(other.0, self.0);
            let eq = _mm256_cmpeq_epi32(self.0, other.0);
            i32x8(_mm256_or_si256(lt, eq))
        }
    }
    #[inline(always)]
    pub(crate) fn simd_gt(self, other: Self) -> i32x8 {
        unsafe { i32x8(_mm256_cmpgt_epi32(self.0, other.0)) }
    }
    #[inline(always)]
    pub(crate) fn simd_ge(self, other: Self) -> i32x8 {
        unsafe {
            let gt = _mm256_cmpgt_epi32(self.0, other.0);
            let eq = _mm256_cmpeq_epi32(self.0, other.0);
            i32x8(_mm256_or_si256(gt, eq))
        }
    }
}

impl i32x8 {
    #[inline(always)]
    fn select_i32(&self, true_val: i32x8, false_val: i32x8) -> i32x8 {
        unsafe { i32x8(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::BitAnd for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_and_si256(self.0, rhs.0)) }
    }
}

impl std::ops::Add for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_add_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_mullo_epi32(self.0, rhs.0)) }
    }
}