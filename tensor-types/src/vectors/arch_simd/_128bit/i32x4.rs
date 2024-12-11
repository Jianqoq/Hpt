use crate::traits::{ Init, SimdSelect, VecTrait };
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 4 i32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct i32x4(pub(crate) __m128i);

impl PartialEq for i32x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi32(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i32x4 {
    fn default() -> Self {
        unsafe { i32x4(_mm_setzero_si128()) }
    }
}

impl VecTrait<i32> for i32x4 {
    const SIZE: usize = 4;
    type Base = i32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i32]) {
        unsafe { _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i32x4(_mm_add_epi32(self.0, _mm_mullo_epi32(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> i32 {
        unsafe {
            let arr: [i32; 4] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
}
impl Init<i32> for i32x4 {
    fn splat(val: i32) -> i32x4 {
        unsafe { i32x4(_mm_set1_epi32(val)) }
    }
}
impl SimdSelect<i32x4> for crate::vectors::arch_simd::_128bit::u32x4::u32x4 {
    fn select(&self, true_val: i32x4, false_val: i32x4) -> i32x4 {
        unsafe { i32x4(_mm_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for i32x4 {
    type Output = i32x4;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(_mm_add_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i32x4 {
    type Output = i32x4;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(_mm_sub_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i32x4 {
    type Output = i32x4;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(_mm_mullo_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i32x4 {
    type Output = i32x4;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 4] = std::mem::transmute(self.0);
            let arr2: [i32; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] / arr2[i];
            }
            i32x4(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for i32x4 {
    type Output = i32x4;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 4] = std::mem::transmute(self.0);
            let arr2: [i32; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] % arr2[i];
            }
            i32x4(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Neg for i32x4 {
    type Output = i32x4;
    fn neg(self) -> Self::Output {
        unsafe { i32x4(_mm_sub_epi32(_mm_setzero_si128(), self.0)) }
    }
}

