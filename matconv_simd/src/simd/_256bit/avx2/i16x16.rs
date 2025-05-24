
use std::arch::x86_64::*;
use crate::simd::_256bit::common::i16x16::i16x16;

impl i16x16 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i16x16(_mm256_add_epi16(self.0, _mm256_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: i16) -> i16x16 {
        unsafe { i16x16(_mm256_set1_epi16(val)) }
    }
}


impl i16x16 {
    #[inline(always)]
    pub(crate) fn simd_ne(self, other: Self) -> i16x16 {
        unsafe {
            let eq = _mm256_cmpeq_epi16(self.0, other.0);
            i16x16(_mm256_xor_si256(eq, _mm256_set1_epi16(-1)))
        }
    }
    #[inline(always)]
    pub(crate) fn simd_lt(self, other: Self) -> i16x16 {
        unsafe { i16x16(_mm256_cmpgt_epi16(other.0, self.0)) }
    }
    #[inline(always)]
    pub(crate) fn simd_le(self, other: Self) -> i16x16 {
        unsafe {
            let lt = _mm256_cmpgt_epi16(other.0, self.0);
            let eq = _mm256_cmpeq_epi16(self.0, other.0);
            i16x16(_mm256_or_si256(lt, eq))
        }
    }
    #[inline(always)]
    pub(crate) fn simd_gt(self, other: Self) -> i16x16 {
        unsafe { i16x16(_mm256_cmpgt_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    pub(crate) fn simd_ge(self, other: Self) -> i16x16 {
        unsafe {
            let gt = _mm256_cmpgt_epi16(self.0, other.0);
            let eq = _mm256_cmpeq_epi16(self.0, other.0);
            i16x16(_mm256_or_si256(gt, eq))
        }
    }
}

impl std::ops::Add for i16x16 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i16x16 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_mullo_epi16(self.0, rhs.0)) }
    }
}