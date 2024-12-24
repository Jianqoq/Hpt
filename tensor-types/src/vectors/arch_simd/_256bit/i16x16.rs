use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::u16x16::u16x16;

/// a vector of 16 i16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct i16x16(pub(crate) __m256i);

impl PartialEq for i16x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi16(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}
impl Default for i16x16 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i16x16(_mm256_setzero_si256()) }
    }
}
impl VecTrait<i16> for i16x16 {
    const SIZE: usize = 16;
    type Base = i16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i16]) {
        unsafe {
            _mm256_storeu_si256(
                &mut self.0,
                _mm256_loadu_si256(slice.as_ptr() as *const __m256i),
            )
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i16x16(_mm256_add_epi16(self.0, _mm256_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> i16 {
        unsafe {
            let arr: [i16; 16] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: i16) -> i16x16 {
        unsafe { i16x16(_mm256_set1_epi16(val)) }
    }
}

impl i16x16 {
    /// convert to array
    #[allow(unused)]
    pub fn as_array(&self) -> [i16; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for i16x16 {
    type SimdMask = i16x16;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i16x16 {
        unsafe { i16x16(_mm256_cmpeq_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i16x16 {
        unsafe {
            let eq = _mm256_cmpeq_epi16(self.0, other.0);
            i16x16(_mm256_xor_si256(eq, _mm256_set1_epi16(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i16x16 {
        unsafe { i16x16(_mm256_cmpgt_epi16(other.0, self.0)) }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i16x16 {
        unsafe {
            let lt = _mm256_cmpgt_epi16(other.0, self.0);
            let eq = _mm256_cmpeq_epi16(self.0, other.0);
            i16x16(_mm256_or_si256(lt, eq))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i16x16 {
        unsafe { i16x16(_mm256_cmpgt_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i16x16 {
        unsafe {
            let gt = _mm256_cmpgt_epi16(self.0, other.0);
            let eq = _mm256_cmpeq_epi16(self.0, other.0);
            i16x16(_mm256_or_si256(gt, eq))
        }
    }
}

impl SimdSelect<i16x16> for crate::vectors::arch_simd::_256bit::i16x16::i16x16 {
    #[inline(always)]
    fn select(&self, true_val: i16x16, false_val: i16x16) -> i16x16 {
        unsafe { i16x16(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for i16x16 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i16x16 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_sub_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i16x16 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i16x16 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i16; 16] = std::mem::transmute(self.0);
            let arr2: [i16; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [i16; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] / arr2[i];
            }
            i16x16(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for i16x16 {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i16; 16] = std::mem::transmute(self.0);
            let arr2: [i16; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [i16; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] % arr2[i];
            }
            i16x16(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Neg for i16x16 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i16x16(_mm256_sign_epi16(self.0, _mm256_set1_epi16(-1))) }
    }
}
impl std::ops::BitAnd for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i16x16(_mm256_xor_si256(self.0, _mm256_set1_epi16(-1))) }
    }
}
impl std::ops::Shl for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i16; 16] = std::mem::transmute(self.0);
            let b: [i16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            i16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i16; 16] = std::mem::transmute(self.0);
            let b: [i16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}

impl SimdMath<i16> for i16x16 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { i16x16(_mm256_max_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { i16x16(_mm256_min_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn relu(self) -> Self {
        unsafe { i16x16(_mm256_max_epi16(self.0, _mm256_setzero_si256())) }
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        unsafe { i16x16(_mm256_min_epi16(self.relu().0, _mm256_set1_epi16(6))) }
    }
}

impl VecConvertor for i16x16 {
    #[inline(always)]
    fn to_i16(self) -> i16x16 {
        self
    }
    #[inline(always)]
    fn to_u16(self) -> u16x16 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f16(self) -> super::f16x16::f16x16 {
        let mut result = [half::f16::ZERO; 16];
        let arr: [i16; 16] = unsafe { std::mem::transmute(self.0) };
        for i in 0..16 {
            result[i] = half::f16::from_f32(arr[i] as f32);
        }
        super::f16x16::f16x16(result)
    }
}
