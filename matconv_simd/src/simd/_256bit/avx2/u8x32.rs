
use std::arch::x86_64::*;
use crate::simd::_256bit::common::u8x32::u8x32;

impl u8x32 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let mut res = [0u8; 32];
            let x: [u8; 32] = std::mem::transmute(self.0);
            let y: [u8; 32] = std::mem::transmute(a.0);
            let z: [u8; 32] = std::mem::transmute(b.0);
            for i in 0..32 {
                res[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            u8x32(_mm256_loadu_si256(res.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    pub(crate) fn splat(val: u8) -> u8x32 {
        unsafe { u8x32(_mm256_set1_epi8(val as i8)) }
    }
}

impl std::ops::Add for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { u8x32(_mm256_add_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [u8; 32] = std::mem::transmute(self.0);
            let b: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0u8; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_mul(b[i]);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}

impl std::ops::BitAnd for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { u8x32(_mm256_and_si256(self.0, rhs.0)) }
    }
}

impl std::ops::BitOr for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { u8x32(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { u8x32(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { u8x32(_mm256_xor_si256(self.0, _mm256_set1_epi8(-1))) }
    }
}
impl std::ops::Shl for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        unsafe {
            let a: [u8; 32] = std::mem::transmute(self.0);
            let b: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        unsafe {
            let a: [u8; 32] = std::mem::transmute(self.0);
            let b: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}