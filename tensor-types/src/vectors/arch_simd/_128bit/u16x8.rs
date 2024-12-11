use crate::traits::{ Init, SimdCompare, SimdSelect, VecTrait };

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 8 u16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u16x8(pub(crate) __m128i);

impl PartialEq for u16x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi16(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u16x8 {
    fn default() -> Self {
        unsafe { u16x8(_mm_setzero_si128()) }
    }
}

impl VecTrait<u16> for u16x8 {
    const SIZE: usize = 8;
    type Base = u16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u16]) {
        unsafe { _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u16x8(_mm_add_epi16(self.0, _mm_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
}

impl SimdSelect<u16x8> for u16x8 {
    fn select(&self, true_val: u16x8, false_val: u16x8) -> u16x8 {
        unsafe { u16x8(_mm_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}
impl Init<u16> for u16x8 {
    fn splat(val: u16) -> u16x8 {
        unsafe { u16x8(_mm_set1_epi16(val as i16)) }
    }
}
impl std::ops::Add for u16x8 {
    type Output = u16x8;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u16x8 {
    type Output = u16x8;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_sub_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u16x8 {
    type Output = u16x8;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u16x8 {
    type Output = u16x8;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] / arr2[i];
            }
            u16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for u16x8 {
    type Output = u16x8;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            u16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}

impl std::ops::BitAnd for u16x8 {
    type Output = u16x8;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_and_si128(self.0, rhs.0)) }
    }
}

impl SimdCompare for u16x8 {
    type SimdMask = u16x8;

    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe { u16x8(_mm_cmpeq_epi16(self.0, other.0)) }
    }

    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe { u16x8(_mm_xor_si128(_mm_cmpeq_epi16(self.0, other.0), _mm_set1_epi16(-1))) }
    }

    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe { u16x8(_mm_cmplt_epi16(self.0, other.0)) }
    }

    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe { u16x8(_mm_xor_si128(_mm_cmpgt_epi16(self.0, other.0), _mm_set1_epi16(-1))) }
    }

    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe { u16x8(_mm_cmpgt_epi16(self.0, other.0)) }
    }

    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe { u16x8(_mm_xor_si128(_mm_cmplt_epi16(self.0, other.0), _mm_set1_epi16(-1))) }
    }
}
