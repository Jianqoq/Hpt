use std::ops::{ Index, IndexMut };

use crate::vectors::traits::{ Init, VecTrait };
use std::arch::x86_64::*;

/// a vector of 4 u64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u64x4(pub(crate) __m256i);

impl PartialEq for u64x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi64(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u64x4 {
    fn default() -> Self {
        unsafe { u64x4(_mm256_setzero_si256()) }
    }
}

impl VecTrait<u64> for u64x4 {
    const SIZE: usize = 4;
    type Base = u64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let x: [u64; 4] = std::mem::transmute(self.0);
            let y: [u64; 4] = std::mem::transmute(a.0);
            let z: [u64; 4] = std::mem::transmute(b.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u64]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn sum(&self) -> u64 {
        unsafe {
            let array: [u64; 4] = std::mem::transmute(self.0);
            array.iter().sum()
        }
    }
}

impl Init<u64> for u64x4 {
    fn splat(val: u64) -> u64x4 {
        u64x4(unsafe { _mm256_set1_epi64x(val as i64) })
    }
}
impl Index<usize> for u64x4 {
    type Output = u64;
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < 4, "Index out of bounds for u64x4");
        unsafe { &*self.as_ptr().add(index) }
    }
}
impl IndexMut<usize> for u64x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < 4, "Index out of bounds for u64x4");
        unsafe { &mut *self.as_mut_ptr().add(index) }
    }
}
impl std::ops::Add for u64x4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        u64x4(unsafe { _mm256_add_epi64(self.0, rhs.0) })
    }
}
impl std::ops::Sub for u64x4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        u64x4(unsafe { _mm256_sub_epi64(self.0, rhs.0) })
    }
}
impl std::ops::Mul for u64x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let x: [u64; 4] = std::mem::transmute(self.0);
            let y: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]);
            }
            u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Div for u64x4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let x: [u64; 4] = std::mem::transmute(self.0);
            let y: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = x[i] / y[i];
            }
            u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u64x4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let x: [u64; 4] = std::mem::transmute(self.0);
            let y: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = x[i] % y[i];
            }
            u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
