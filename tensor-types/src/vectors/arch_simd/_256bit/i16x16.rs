use crate::vectors::traits::VecTrait;
use std::arch::x86_64::*;
/// a vector of 16 i16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct i16x16(pub(crate) __m256i);

impl PartialEq for i16x16 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi16(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i16x16 {
    fn default() -> Self {
        unsafe { i16x16(_mm256_setzero_si256()) }
    }
}

impl VecTrait<i16> for i16x16 {
    const SIZE: usize = 16;
    type Base = i16;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i16x16(_mm256_add_epi16(_mm256_mullo_epi16(self.0, a.0), b.0)) }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i16]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn sum(&self) -> i16 {
        unsafe {
            let a: [i16; 16] = std::mem::transmute(self.0);
            a.iter().sum()
        }
    }
    fn splat(val: i16) -> i16x16 {
        i16x16(unsafe { _mm256_set1_epi16(val) })
    }
}

impl std::ops::Add for i16x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i16x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_sub_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i16x16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i16x16(_mm256_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i16x16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i16; 16] = std::mem::transmute(self.0);
            let b: [i16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i16; 16];
            for i in 0..16 {
                result[i] = a[i] / b[i];
            }
            i16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for i16x16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i16; 16] = std::mem::transmute(self.0);
            let b: [i16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i16; 16];
            for i in 0..16 {
                result[i] = a[i] % b[i];
            }
            i16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
