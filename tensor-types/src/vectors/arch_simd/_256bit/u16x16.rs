use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i16x16::i16x16;

/// a vector of 16 u16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u16x16(pub(crate) __m256i);

impl PartialEq for u16x16 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi16(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u16x16 {
    fn default() -> Self {
        unsafe { u16x16(_mm256_setzero_si256()) }
    }
}

impl VecTrait<u16> for u16x16 {
    const SIZE: usize = 16;
    type Base = u16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u16]) {
        unsafe {
            _mm256_storeu_si256(
                &mut self.0,
                _mm256_loadu_si256(slice.as_ptr() as *const __m256i),
            )
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u16x16(_mm256_add_epi16(self.0, _mm256_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    fn splat(val: u16) -> u16x16 {
        unsafe { u16x16(_mm256_set1_epi16(val as i16)) }
    }
}

impl u16x16 {
    #[allow(unused)]
    fn as_array(&self) -> [u16; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdSelect<u16x16> for u16x16 {
    fn select(&self, true_val: u16x16, false_val: u16x16) -> u16x16 {
        unsafe { u16x16(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for u16x16 {
    type Output = u16x16;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u16x16 {
    type Output = u16x16;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_sub_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u16x16 {
    type Output = u16x16;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u16x16 {
    type Output = u16x16;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            let arr2: [u16; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] / arr2[i];
            }
            u16x16(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u16x16 {
    type Output = u16x16;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            let arr2: [u16; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] % arr2[i];
            }
            u16x16(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}

impl std::ops::BitAnd for u16x16 {
    type Output = u16x16;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u16x16 {
    type Output = u16x16;
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u16x16 {
    type Output = u16x16;
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u16x16 {
    type Output = Self;
    fn not(self) -> Self::Output {
        unsafe { u16x16(_mm256_xor_si256(self.0, _mm256_set1_epi16(-1))) }
    }
}
impl std::ops::Shl for u16x16 {
    type Output = Self;
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
impl SimdCompare for u16x16 {
    type SimdMask = i16x16;

    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }

    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }

    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }

    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }

    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }

    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl SimdMath<u16> for u16x16 {
    fn max(self, other: Self) -> Self {
        unsafe { u16x16(_mm256_max_epi16(self.0, other.0)) }
    }
    fn min(self, other: Self) -> Self {
        unsafe { u16x16(_mm256_min_epi16(self.0, other.0)) }
    }
    fn relu(self) -> Self {
        unsafe { u16x16(_mm256_max_epi16(self.0, _mm256_setzero_si256())) }
    }
    fn relu6(self) -> Self {
        unsafe { u16x16(_mm256_min_epi16(self.relu().0, _mm256_set1_epi16(6))) }
    }
}

impl VecConvertor for u16x16 {
    fn to_u16(self) -> u16x16 {
        self
    }
    fn to_i16(self) -> i16x16 {
        unsafe { std::mem::transmute(self) }
    }
    fn to_f16(self) -> super::f16x16::f16x16 {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            let mut result = [half::f16::ZERO; 16];
            for i in 0..16 {
                result[i] = half::f16::from_f32(arr[i] as f32);
            }
            super::f16x16::f16x16(result)
        }
    }
    fn to_bf16(self) -> super::bf16x16::bf16x16 {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            let mut result = [half::bf16::ZERO; 16];
            for i in 0..16 {
                result[i] = half::bf16::from_f32(arr[i] as f32);
            }
            super::bf16x16::bf16x16(result)
        }
    }
}
