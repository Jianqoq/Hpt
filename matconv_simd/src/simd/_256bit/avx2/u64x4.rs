
use std::arch::x86_64::*;
use crate::simd::_256bit::common::u64x4::u64x4;

impl u64x4 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(a.0);
            let arr3: [u64; 4] = std::mem::transmute(b.0);
            let mut arr4: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            u64x4(_mm256_loadu_si256(arr4.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    pub(crate) fn splat(val: u64) -> u64x4 {
        unsafe { u64x4(_mm256_set1_epi64x(val as i64)) }
    }
}

impl std::ops::Add for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_add_epi64(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] * arr2[i];
            }
            u64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}