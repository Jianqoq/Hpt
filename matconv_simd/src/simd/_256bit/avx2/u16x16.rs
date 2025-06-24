
use std::arch::x86_64::*;
use crate::simd::_256bit::common::i16x16::i16x16;
use crate::simd::_256bit::common::u16x16::u16x16;

impl u16x16 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u16x16(_mm256_add_epi16(self.0, _mm256_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: u16) -> u16x16 {
        unsafe { u16x16(_mm256_set1_epi16(val as i16)) }
    }
}

impl std::ops::Add for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_mullo_epi16(self.0, rhs.0)) }
    }
}

impl std::ops::BitAnd for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u16x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { u16x16(_mm256_xor_si256(self.0, _mm256_set1_epi16(-1))) }
    }
}
impl std::ops::Shl for u16x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 16] = std::mem::transmute(self.0);
            let b: [u16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for u16x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 16] = std::mem::transmute(self.0);
            let b: [u16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            u16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}