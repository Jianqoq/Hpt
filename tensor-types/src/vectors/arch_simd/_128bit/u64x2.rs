use crate::traits::{ Init, SimdSelect, VecTrait };

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 2 u64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u64x2(pub(crate) __m128i);

impl PartialEq for u64x2 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi64(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u64x2 {
    fn default() -> Self {
        unsafe { u64x2(_mm_setzero_si128()) }
    }
}

impl VecTrait<u64> for u64x2 {
    const SIZE: usize = 2;
    type Base = u64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u64]) {
        unsafe { _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [u64; 2] = std::mem::transmute(self.0);
            let arr2: [u64; 2] = std::mem::transmute(a.0);
            let arr3: [u64; 2] = std::mem::transmute(b.0);
            let mut arr4: [u64; 2] = [0; 2];
            for i in 0..2 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            u64x2(_mm_loadu_si128(arr4.as_ptr() as *const __m128i))
        }
    }
    #[inline(always)]
    fn sum(&self) -> u64 {
        unsafe {
            let arr: [u64; 2] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
}
impl Init<u64> for u64x2 {
    fn splat(val: u64) -> u64x2 {
        unsafe { u64x2(_mm_set1_epi64x(val as i64)) }
    }
}
impl SimdSelect<u64x2> for u64x2 {
    fn select(&self, true_val: u64x2, false_val: u64x2) -> u64x2 {
        unsafe { u64x2(_mm_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for u64x2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        unsafe { u64x2(_mm_add_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u64x2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        unsafe { u64x2(_mm_sub_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u64x2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 2] = std::mem::transmute(self.0);
            let arr2: [u64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i] * arr2[i];
            }
            u64x2(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Div for u64x2 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 2] = std::mem::transmute(self.0);
            let arr2: [u64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i] / arr2[i];
            }
            u64x2(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for u64x2 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 2] = std::mem::transmute(self.0);
            let arr2: [u64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i] % arr2[i];
            }
            u64x2(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
