use crate::{convertion::VecConvertor, traits::{ SimdCompare, SimdMath, VecTrait }};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i8x16::i8x16;

/// a vector of 16 u8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u8x16(pub(crate) __m128i);

impl PartialEq for u8x16 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi8(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u8x16 {
    fn default() -> Self {
        unsafe { u8x16(_mm_setzero_si128()) }
    }
}

impl VecTrait<u8> for u8x16 {
    const SIZE: usize = 16;
    type Base = u8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u8]) {
        unsafe { _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u8x16(_mm_add_epi8(self.0, _mm_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> u8 {
        unsafe {
            let x: [u8; 16] = std::mem::transmute(self.0);
            x.iter().sum()
        }
    }
    fn splat(val: u8) -> u8x16 {
        unsafe { u8x16(_mm_set1_epi8(val as i8)) }
    }
}

impl u8x16 {
    #[allow(unused)]
    fn as_array(&self) -> [u8; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for u8x16 {
    type SimdMask = i8x16;
    fn simd_eq(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }
    fn simd_ne(self, other: Self) -> i8x16 {
        unsafe { 
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }
    fn simd_lt(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }
    fn simd_le(self, other: Self) -> i8x16 {
        unsafe { 
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }
    fn simd_gt(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }
    fn simd_ge(self, other: Self) -> i8x16 {
        unsafe { 
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl std::ops::Add for u8x16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        unsafe { u8x16(_mm_add_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u8x16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        unsafe { u8x16(_mm_sub_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u8x16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        unsafe { u8x16(_mm_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u8x16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 16] = std::mem::transmute(self.0);
            let arr2: [u8; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] / arr2[i];
            }
            u8x16(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for u8x16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 16] = std::mem::transmute(self.0);
            let arr2: [u8; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] % arr2[i];
            }
            u8x16(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}

impl std::ops::BitAnd for u8x16 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        unsafe { u8x16(_mm_and_si128(self.0, rhs.0)) }
    }
}

impl std::ops::BitOr for u8x16 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        unsafe { u8x16(_mm_or_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u8x16 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { u8x16(_mm_xor_si128(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u8x16 {
    type Output = Self;
    fn not(self) -> Self {
        unsafe { u8x16(_mm_xor_si128(self.0, _mm_set1_epi8(-1))) }
    }
}
impl std::ops::Shl for u8x16 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self {
        unsafe {
            let a: [u8; 16] = std::mem::transmute(self.0);
            let b: [u8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] << b[i];
            }
            u8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Shr for u8x16 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self {
        unsafe {
            let a: [u8; 16] = std::mem::transmute(self.0);
            let b: [u8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] >> b[i];
            }
            u8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}

impl SimdMath<u8> for u8x16 {
    fn max(self, other: Self) -> Self {
        unsafe { u8x16(_mm_max_epi8(self.0, other.0)) }
    }
    fn min(self, other: Self) -> Self {
        unsafe { u8x16(_mm_min_epi8(self.0, other.0)) }
    }
    fn relu(self) -> Self {
        unsafe { u8x16(_mm_max_epi8(self.0, _mm_setzero_si128())) }
    }
    fn relu6(self) -> Self {
        unsafe { u8x16(_mm_min_epi8(self.relu().0, _mm_set1_epi8(6))) }
    }
}

impl VecConvertor for u8x16 {
}