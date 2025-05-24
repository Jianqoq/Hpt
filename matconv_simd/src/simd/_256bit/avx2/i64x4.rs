
use std::arch::x86_64::*;
use crate::simd::_256bit::common::i64x4::i64x4;

impl i64x4 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let arr2: [i64; 4] = std::mem::transmute(a.0);
            let arr3: [i64; 4] = std::mem::transmute(b.0);
            let mut arr4: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            i64x4(_mm256_loadu_si256(arr4.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    pub(crate) fn splat(val: i64) -> i64x4 {
        unsafe { i64x4(_mm256_set1_epi64x(val)) }
    }
}

impl std::ops::Add for i64x4 {
    type Output = i64x4;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i64x4(_mm256_add_epi64(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for i64x4 {
    type Output = i64x4;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let arr2: [i64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] * arr2[i];
            }
            i64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}