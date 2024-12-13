use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::u8x32::u8x32;

/// a vector of 32 i8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct i8x32(pub(crate) __m256i);

impl PartialEq for i8x32 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi8(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i8x32 {
    fn default() -> Self {
        unsafe { i8x32(_mm256_setzero_si256()) }
    }
}

impl VecTrait<i8> for i8x32 {
    const SIZE: usize = 32;
    type Base = i8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i8]) {
        unsafe {
            _mm256_storeu_si256(
                &mut self.0,
                _mm256_loadu_si256(slice.as_ptr() as *const __m256i),
            );
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i8x32(_mm256_add_epi8(_mm256_mullo_epi16(self.0, a.0), b.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        unsafe {
            let sum = _mm256_sad_epu8(self.0, _mm256_setzero_si256());
            _mm256_cvtsi256_si32(sum) as i8
        }
    }
    fn splat(val: i8) -> i8x32 {
        unsafe { i8x32(_mm256_set1_epi8(val)) }
    }
}

impl i8x32 {
    #[allow(unused)]
    fn as_array(&self) -> [i8; 32] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for i8x32 {
    type SimdMask = i8x32;
    fn simd_eq(self, other: Self) -> i8x32 {
        unsafe { i8x32(_mm256_cmpeq_epi8(self.0, other.0)) }
    }
    fn simd_ne(self, other: Self) -> i8x32 {
        unsafe {
            let eq = _mm256_cmpeq_epi8(self.0, other.0);
            i8x32(_mm256_xor_si256(eq, _mm256_set1_epi8(-1)))
        }
    }
    fn simd_lt(self, other: Self) -> i8x32 {
        unsafe { i8x32(_mm256_cmpgt_epi8(other.0, self.0)) }
    }
    fn simd_le(self, other: Self) -> i8x32 {
        unsafe {
            let gt = _mm256_cmpgt_epi8(self.0, other.0);
            i8x32(_mm256_xor_si256(gt, _mm256_set1_epi8(-1)))
        }
    }
    fn simd_gt(self, other: Self) -> i8x32 {
        unsafe { i8x32(_mm256_cmpgt_epi8(self.0, other.0)) }
    }
    fn simd_ge(self, other: Self) -> i8x32 {
        unsafe {
            let gt = _mm256_cmpgt_epi8(self.0, other.0);
            let eq = _mm256_cmpeq_epi8(self.0, other.0);
            i8x32(_mm256_or_si256(gt, eq))
        }
    }
}

impl std::ops::Add for i8x32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_add_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i8x32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_sub_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i8x32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i8x32 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i] / b[i];
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for i8x32 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i] % b[i];
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Neg for i8x32 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        unsafe { i8x32(_mm256_sub_epi8(_mm256_setzero_si256(), self.0)) }
    }
}
impl std::ops::BitAnd for i8x32 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i8x32 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i8x32 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i8x32 {
    type Output = Self;
    fn not(self) -> Self::Output {
        unsafe { i8x32(_mm256_xor_si256(self.0, _mm256_set1_epi8(-1))) }
    }
}
impl std::ops::Shl for i8x32 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i] << b[i];
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for i8x32 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i] >> b[i];
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}

impl SimdMath<i8> for i8x32 {
    fn max(self, other: Self) -> Self {
        unsafe { i8x32(_mm256_max_epi8(self.0, other.0)) }
    }
    fn min(self, other: Self) -> Self {
        unsafe { i8x32(_mm256_min_epi8(self.0, other.0)) }
    }
    fn relu(self) -> Self {
        unsafe { i8x32(_mm256_max_epi8(self.0, _mm256_setzero_si256())) }
    }
    fn relu6(self) -> Self {
        unsafe { i8x32(_mm256_min_epi8(self.relu().0, _mm256_set1_epi8(6))) }
    }
}

impl VecConvertor for i8x32 {
    fn to_i8(self) -> i8x32 {
        self
    }
    fn to_u8(self) -> u8x32 {
        unsafe { std::mem::transmute(self) }
    }
    fn to_bool(self) -> super::boolx32::boolx32 {
        unsafe { std::mem::transmute(self) }
    }
}

