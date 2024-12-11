use crate::vectors::traits::VecTrait;
use std::arch::x86_64::*;

/// a vector of 8 u32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u32x8(pub(crate) __m256i);

impl PartialEq for u32x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi32(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u32x8 {
    fn default() -> Self {
        u32x8(unsafe { _mm256_setzero_si256() })
    }
}

impl VecTrait<u32> for u32x8 {
    const SIZE: usize = 8;
    type Base = u32;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        u32x8(unsafe { _mm256_add_epi32(_mm256_mullo_epi32(self.0, a.0), b.0) })
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u32]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn sum(&self) -> u32 {
        unsafe {
            let array: [u32; 8] = std::mem::transmute(self.0);
            array.iter().sum()
        }
    }
    fn splat(val: u32) -> u32x8 {
        u32x8(unsafe { _mm256_set1_epi32(val as i32) })
    }
}

impl std::ops::Add for u32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        u32x8(unsafe { _mm256_add_epi32(self.0, rhs.0) })
    }
}
impl std::ops::Sub for u32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        u32x8(unsafe { _mm256_sub_epi32(self.0, rhs.0) })
    }
}
impl std::ops::Mul for u32x8 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        u32x8(unsafe { _mm256_mullo_epi32(self.0, rhs.0) })
    }
}
impl std::ops::Div for u32x8 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u32; 8] = std::mem::transmute(self.0);
            let b: [u32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0u32; 8];
            for i in 0..8 {
                result[i] = a[i] / b[i];
            }
            u32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u32x8 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u32; 8] = std::mem::transmute(self.0);
            let b: [u32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0u32; 8];
            for i in 0..8 {
                result[i] = a[i] % b[i];
            }
            u32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
