use crate::traits::{ Init, VecTrait };

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 16 i8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct i8x16(pub(crate) __m128i);

impl PartialEq for i8x16 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi8(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i8x16 {
    fn default() -> Self {
        unsafe { i8x16(_mm_setzero_si128()) }
    }
}

impl VecTrait<i8> for i8x16 {
    const SIZE: usize = 16;
    type Base = i8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i8]) {
        unsafe {
            _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i));
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i8x16(_mm_add_epi8(_mm_mullo_epi16(self.0, a.0), b.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        unsafe {
            let sum = _mm_sad_epu8(self.0, _mm_setzero_si128());
            _mm_cvtsi128_si32(sum) as i8
        }
    }
}
impl Init<i8> for i8x16 {
    fn splat(val: i8) -> i8x16 {
        unsafe { i8x16(_mm_set1_epi8(val)) }
    }
}
impl std::ops::Add for i8x16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_add_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i8x16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_sub_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i8x16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i8x16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] / b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for i8x16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] % b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Neg for i8x16 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        unsafe { i8x16(_mm_sub_epi8(_mm_setzero_si128(), self.0)) }
    }
}
