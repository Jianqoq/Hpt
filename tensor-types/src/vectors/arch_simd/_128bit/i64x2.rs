use crate::traits::{ Init, SimdSelect, VecTrait };

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 2 i64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct i64x2(pub(crate) __m128i);

impl PartialEq for i64x2 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi64(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i64x2 {
    fn default() -> Self {
        unsafe { i64x2(_mm_setzero_si128()) }
    }
}

impl VecTrait<i64> for i64x2 {
    const SIZE: usize = 2;
    type Base = i64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i64]) {
        unsafe { _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(a.0);
            let arr3: [i64; 2] = std::mem::transmute(b.0);
            let mut arr4: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            i64x2(_mm_loadu_si128(arr4.as_ptr() as *const __m128i))
        }
    }
    #[inline(always)]
    fn sum(&self) -> i64 {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
}

impl SimdSelect<i64x2> for crate::vectors::arch_simd::_128bit::u64x2::u64x2 {
    fn select(&self, true_val: i64x2, false_val: i64x2) -> i64x2 {
        unsafe { i64x2(_mm_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}
impl Init<i64> for i64x2 {
    fn splat(val: i64) -> i64x2 {
        unsafe { i64x2(_mm_set1_epi64x(val)) }
    }
}
impl std::ops::Add for i64x2 {
    type Output = i64x2;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(_mm_add_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i64x2 {
    type Output = i64x2;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(_mm_sub_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i64x2 {
    type Output = i64x2;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i] * arr2[i];
            }
            i64x2(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Div for i64x2 {
    type Output = i64x2;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i] / arr2[i];
            }
            i64x2(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for i64x2 {
    type Output = i64x2;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i] % arr2[i];
            }
            i64x2(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Neg for i64x2 {
    type Output = i64x2;
    fn neg(self) -> Self::Output {
        unsafe { i64x2(_mm_sub_epi64(_mm_setzero_si128(), self.0)) }
    }
}

