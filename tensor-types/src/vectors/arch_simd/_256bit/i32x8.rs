use crate::vectors::traits::VecTrait;
use std::arch::x86_64::*;

/// a vector of 8 i32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct i32x8(pub(crate) __m256i);

impl Default for i32x8 {
    fn default() -> Self {
        unsafe { i32x8(_mm256_setzero_si256()) }
    }
}

impl PartialEq for i32x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi32(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl VecTrait<i32> for i32x8 {
    const SIZE: usize = 8;
    type Base = i32;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i32x8(_mm256_add_epi32(_mm256_mullo_epi32(self.0, a.0), b.0)) }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i32]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn sum(&self) -> i32 {
        unsafe {
            let mut sum = 0;
            for i in 0..8 {
                sum += *self.as_ptr().add(i);
            }
            sum
        }
    }
    fn splat(val: i32) -> i32x8 {
        unsafe { i32x8(_mm256_set1_epi32(val)) }
    }
}

impl std::ops::Add for i32x8 {
    type Output = i32x8;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_add_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i32x8 {
    type Output = i32x8;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_sub_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i32x8 {
    type Output = i32x8;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_mullo_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i32x8 {
    type Output = i32x8;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 8] = std::mem::transmute(self.0);
            let b: [i32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = a[i] / b[i];
            }
            i32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for i32x8 {
    type Output = i32x8;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 8] = std::mem::transmute(self.0);
            let b: [i32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = a[i] % b[i];
            }
            i32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
