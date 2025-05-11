
use std::arch::x86_64::*;
use crate::simd::_256bit::common::i8x32::i8x32;

impl i8x32 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let mut res = [0i8; 32];
            let x: [i8; 32] = std::mem::transmute(self.0);
            let y: [i8; 32] = std::mem::transmute(a.0);
            let z: [i8; 32] = std::mem::transmute(b.0);
            for i in 0..32 {
                res[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            i8x32(_mm256_loadu_si256(res.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    pub(crate) fn splat(val: i8) -> i8x32 {
        unsafe { i8x32(_mm256_set1_epi8(val)) }
    }
}

impl std::ops::Add for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i8x32(_mm256_add_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 32] = std::mem::transmute(self.0);
            let b: [i8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_mul(b[i]);
            }
            i8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}