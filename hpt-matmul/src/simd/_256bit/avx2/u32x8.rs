
use std::arch::x86_64::*;
use crate::simd::_256bit::common::i32x8::i32x8;
use crate::simd::_256bit::common::u32x8::u32x8;

impl u32x8 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u32x8(_mm256_add_epi32(self.0, _mm256_mullo_epi32(a.0, b.0))) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: u32) -> u32x8 {
        unsafe { u32x8(_mm256_set1_epi32(val as i32)) }
    }
}

impl i32x8 {
    #[inline(always)]
    pub(crate) fn select_u32(&self, true_val: u32x8, false_val: u32x8) -> u32x8 {
        unsafe { u32x8(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Sub for u32x8 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_sub_epi32(self.0, rhs.0)) }
    }
}

impl std::ops::BitAnd for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { u32x8(_mm256_xor_si256(self.0, _mm256_set1_epi32(-1))) }
    }
}
impl std::ops::Shl for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u32; 8] = std::mem::transmute(self.0);
            let b: [u32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u32; 8] = std::mem::transmute(self.0);
            let b: [u32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            u32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}

impl std::ops::Add for u32x8 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_add_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_mullo_epi32(self.0, rhs.0)) }
    }
}