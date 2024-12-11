use crate::vectors::traits::{ Init, VecTrait };
use std::arch::x86_64::*;

/// a vector of 32 i8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct i8x32(pub(crate) __m256i);

impl PartialEq for i8x32 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi8(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i8x32 {
    fn default() -> Self {
        unsafe { i8x32(_mm256_setzero_si256()) }
    }
}

impl VecTrait<i8> for i8x32 {
    const SIZE: usize = 32;
    type Base = i8;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let x: [i8; 32] = std::mem::transmute(self.0);
            let y: [i8; 32] = std::mem::transmute(a.0);
            let z: [i8; 32] = std::mem::transmute(b.0);
            let mut result = [0i8; 32];
            for i in 0..32 {
                result[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i8]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        unsafe {
            let x: [i8; 32] = std::mem::transmute(self.0);
            x.iter().sum()
        }
    }
}

impl Init<i8> for i8x32 {
    fn splat(val: i8) -> i8x32 {
        unsafe { i8x32(_mm256_set1_epi8(val)) }
    }
}
impl std::ops::Add for i8x32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        i8x32(unsafe { _mm256_add_epi8(self.0, rhs.0) })
    }
}
impl std::ops::Sub for i8x32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        i8x32(unsafe { _mm256_sub_epi8(self.0, rhs.0) })
    }
}
impl std::ops::Mul for i8x32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let x: [i8; 32] = std::mem::transmute(self.0);
            let y: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 32];
            for i in 0..32 {
                result[i] = x[i].wrapping_mul(y[i]);
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Div for i8x32 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 32];
            for i in 0..32 {
                result[i] = a[i] / b[i];
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for i8x32 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 32];
            for i in 0..32 {
                result[i] = a[i] % b[i];
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Neg for i8x32 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        unsafe { i8x32(_mm256_sub_epi8(_mm256_setzero_si256(), self.0)) }
    }
}
