use crate::traits::{ Init, SimdSelect, VecTrait };

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 4 u32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u32x4(pub(crate) __m128i);

impl Default for u32x4 {
    fn default() -> Self {
        unsafe { u32x4(_mm_setzero_si128()) }
    }
}

impl PartialEq for u32x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi32(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}
impl VecTrait<u32> for u32x4 {
    const SIZE: usize = 4;
    type Base = u32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u32]) {
        unsafe { _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u32x4(_mm_add_epi32(self.0, _mm_mullo_epi32(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> u32 {
        unsafe {
            let arr: [u32; 4] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
}

impl SimdSelect<u32x4> for u32x4 {
    fn select(&self, true_val: u32x4, false_val: u32x4) -> u32x4 {
        unsafe { u32x4(_mm_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}
impl Init<u32> for u32x4 {
    fn splat(val: u32) -> u32x4 {
        unsafe { u32x4(_mm_set1_epi32(val as i32)) }
    }
}
impl std::ops::Add for u32x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(_mm_add_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u32x4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(_mm_sub_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u32x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(_mm_mullo_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u32x4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u32; 4] = std::mem::transmute(self.0);
            let arr2: [u32; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [u32; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] / arr2[i];
            }
            u32x4(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for u32x4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u32; 4] = std::mem::transmute(self.0);
            let arr2: [u32; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [u32; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] % arr2[i];
            }
            u32x4(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
