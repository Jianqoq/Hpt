use crate::{convertion::VecConvertor, traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait }};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i32x8::i32x8;

/// a vector of 4 u32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u32x8(pub(crate) __m256i);

impl Default for u32x8 {
    fn default() -> Self {
        unsafe { u32x8(_mm256_setzero_si256()) }
    }
}

impl PartialEq for u32x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi32(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}
impl VecTrait<u32> for u32x8 {
    const SIZE: usize = 8;
    type Base = u32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u32]) {
        unsafe { _mm256_storeu_si256(&mut self.0, _mm256_loadu_si256(slice.as_ptr() as *const __m256i)) }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u32x8(_mm256_add_epi32(self.0, _mm256_mullo_epi32(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> u32 {
        unsafe {
            let arr: [u32; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    fn splat(val: u32) -> u32x8 {
        unsafe { u32x8(_mm256_set1_epi32(val as i32)) }
    }
}

impl u32x8 {
    /// convert the vector to an array
    #[allow(unused)]
    pub fn as_array(&self) -> [u32; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for u32x8 {
    type SimdMask = i32x8;

    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }

    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }

    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }

    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }

    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }

    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl SimdSelect<u32x8> for u32x8 {
    fn select(&self, true_val: u32x8, false_val: u32x8) -> u32x8 {
        unsafe { u32x8(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for u32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_add_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_sub_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u32x8 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_mullo_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u32x8 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u32; 8] = std::mem::transmute(self.0);
            let arr2: [u32; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u32; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] / arr2[i];
            }
            u32x8(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u32x8 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u32; 8] = std::mem::transmute(self.0);
            let arr2: [u32; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u32; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            u32x8(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::BitAnd for u32x8 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u32x8 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u32x8 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u32x8 {
    type Output = Self;
    fn not(self) -> Self::Output {
        unsafe { u32x8(_mm256_xor_si256(self.0, _mm256_set1_epi32(-1))) }
    }
}
impl std::ops::Shl for u32x8 {
    type Output = Self;
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

impl SimdMath<u32> for u32x8 {
    fn max(self, other: Self) -> Self {
        unsafe { u32x8(_mm256_max_epi32(self.0, other.0)) }
    }
    fn min(self, other: Self) -> Self {
        unsafe { u32x8(_mm256_min_epi32(self.0, other.0)) }
    }
    fn relu(self) -> Self {
        unsafe { u32x8(_mm256_max_epi32(self.0, _mm256_setzero_si256())) }
    }
    fn relu6(self) -> Self {
        unsafe { u32x8(_mm256_min_epi32(self.relu().0, _mm256_set1_epi32(6))) }
    }
}

impl VecConvertor for u32x8 {
    fn to_u32(self) -> u32x8 {
        self
    }
    fn to_i32(self) -> i32x8 {
        unsafe { std::mem::transmute(self) }
    }
    fn to_f32(self) -> super::f32x8::f32x8 {
        unsafe {
            let arr: [u32; 8] = std::mem::transmute(self.0);
            let mut result = [0.0f32; 8];
            for i in 0..8 {
                result[i] = arr[i] as f32;
            }
            super::f32x8::f32x8(_mm256_loadu_ps(result.as_ptr()))
        }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_usize(self) -> super::usizex4::usizex4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_isize(self) -> super::isizex4::isizex4 {
        unsafe { std::mem::transmute(self) }
    }
}