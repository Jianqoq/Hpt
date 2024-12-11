use std::ops::{ Index, IndexMut };

use crate::vectors::traits::{ Init, VecTrait };
use std::arch::x86_64::*;
/// a vector of 32 u8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u8x32(pub(crate) __m256i);

impl PartialEq for u8x32 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi8(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u8x32 {
    fn default() -> Self {
        u8x32(unsafe { _mm256_setzero_si256() })
    }
}

impl VecTrait<u8> for u8x32 {
    const SIZE: usize = 32;
    type Base = u8;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let x: [u8; 32] = std::mem::transmute(self.0);
            let y: [u8; 32] = std::mem::transmute(a.0);
            let z: [u8; 32] = std::mem::transmute(b.0);
            let mut result = [0u8; 32];
            for i in 0..32 {
                result[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u8]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn sum(&self) -> u8 {
        unsafe {
            let array: [u8; 32] = std::mem::transmute(self.0);
            array.iter().sum()
        }
    }
}

impl Init<u8> for u8x32 {
    fn splat(val: u8) -> u8x32 {
        u8x32(unsafe { _mm256_set1_epi8(val as i8) })
    }
}
impl Index<usize> for u8x32 {
    type Output = u8;
    fn index(&self, idx: usize) -> &Self::Output {
        assert!(idx < 32, "Index out of bounds for u8x32");
        unsafe { &*self.as_ptr().add(idx) }
    }
}
impl IndexMut<usize> for u8x32 {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        assert!(idx < 32, "Index out of bounds for u8x32");
        unsafe { &mut *self.as_mut_ptr().add(idx) }
    }
}
impl std::ops::Add for u8x32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        u8x32(unsafe { _mm256_add_epi8(self.0, rhs.0) })
    }
}
impl std::ops::Sub for u8x32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        u8x32(unsafe { _mm256_sub_epi8(self.0, rhs.0) })
    }
}
impl std::ops::Mul for u8x32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let x: [u8; 32] = std::mem::transmute(self.0);
            let y: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0u8; 32];
            for i in 0..32 {
                result[i] = x[i].wrapping_mul(y[i]);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Div for u8x32 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let x: [u8; 32] = std::mem::transmute(self.0);
            let y: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0u8; 32];
            for i in 0..32 {
                result[i] = x[i].wrapping_div(y[i]);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u8x32 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let x: [u8; 32] = std::mem::transmute(self.0);
            let y: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0u8; 32];
            for i in 0..32 {
                result[i] = x[i].wrapping_rem(y[i]);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
