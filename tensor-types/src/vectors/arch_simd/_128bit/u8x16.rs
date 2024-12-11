use crate::traits::{ Init, SimdCompare, VecTrait };
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
}
impl Init<u8> for u8x16 {
    fn splat(val: u8) -> u8x16 {
        unsafe { u8x16(_mm_set1_epi8(val as i8)) }
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

impl SimdCompare for u8x16 {
    type SimdMask = Self;

    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe { u8x16(_mm_cmpeq_epi8(self.0, other.0)) }
    }

    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe { u8x16(_mm_xor_si128(_mm_cmpeq_epi8(self.0, other.0), _mm_set1_epi8(-1))) }
    }

    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe { u8x16(_mm_cmplt_epi8(self.0, other.0)) }
    }

    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe { u8x16(_mm_xor_si128(_mm_cmpgt_epi8(self.0, other.0), _mm_set1_epi8(-1))) }
    }

    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe { u8x16(_mm_cmpgt_epi8(self.0, other.0)) }
    }

    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe { u8x16(_mm_xor_si128(_mm_cmplt_epi8(self.0, other.0), _mm_set1_epi8(-1))) }
    }
}
