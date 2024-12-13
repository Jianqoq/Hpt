use crate::{convertion::VecConvertor, traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait }};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::u32x4::u32x4;

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
    fn splat(val: i32) -> i32x4 {
        unsafe { i32x4(_mm_set1_epi32(val)) }
    }
}

impl i32x4 {
    #[allow(unused)]
    fn as_array(&self) -> [i32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for i32x4 {
    type SimdMask = i32x4;
    fn simd_eq(self, other: Self) -> i32x4 {
        unsafe { i32x4(_mm_cmpeq_epi32(self.0, other.0)) }
    }
    fn simd_ne(self, other: Self) -> i32x4 {
        unsafe { 
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            i32x4(_mm_xor_si128(eq, _mm_set1_epi32(-1)))
        }
    }
    fn simd_lt(self, other: Self) -> i32x4 {
        unsafe { i32x4(_mm_cmplt_epi32(self.0, other.0)) }
    }
    fn simd_le(self, other: Self) -> i32x4 {
        unsafe { 
            let lt = _mm_cmplt_epi32(self.0, other.0);
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            i32x4(_mm_or_si128(lt, eq))
        }
    }
    fn simd_gt(self, other: Self) -> i32x4 {
        unsafe { i32x4(_mm_cmpgt_epi32(self.0, other.0)) }
    }
    fn simd_ge(self, other: Self) -> i32x4 {
        unsafe { 
            let gt = _mm_cmpgt_epi32(self.0, other.0);
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            i32x4(_mm_or_si128(gt, eq))
        }
    }
}

impl SimdSelect<i32x4> for crate::vectors::arch_simd::_128bit::i32x4::i32x4 {
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
impl std::ops::BitAnd for i32x4 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(_mm_and_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i32x4 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(_mm_or_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i32x4 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i32x4(_mm_xor_si128(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i32x4 {
    type Output = Self;
    fn not(self) -> Self::Output {
        unsafe { i32x4(_mm_xor_si128(self.0, _mm_set1_epi32(-1))) }
    }
}
impl std::ops::Shl for i32x4 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 4] = std::mem::transmute(self.0);
            let b: [i32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i] << b[i];
            }
            i32x4(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Shr for i32x4 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 4] = std::mem::transmute(self.0);
            let b: [i32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i] >> b[i];
            }
            i32x4(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl SimdMath<i32> for i32x4 {
    fn max(self, other: Self) -> Self {
        unsafe { i32x4(_mm_max_epi32(self.0, other.0)) }
    }
    fn min(self, other: Self) -> Self {
        unsafe { i32x4(_mm_min_epi32(self.0, other.0)) }
    }
    fn relu(self) -> Self {
        unsafe { i32x4(_mm_max_epi32(self.0, _mm_setzero_si128())) }
    }
    fn relu6(self) -> Self {
        unsafe { i32x4(_mm_min_epi32(self.relu().0, _mm_set1_epi32(6))) }
    }
}

impl VecConvertor for i32x4 {
    fn to_i32(self) -> i32x4 {
        self
    }
    fn to_u32(self) -> u32x4 {
        unsafe { std::mem::transmute(self) }
    }
    fn to_f32(self) -> super::f32x4::f32x4 {
        unsafe {
            super::f32x4::f32x4(_mm_cvtepi32_ps(self.0))
        }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_isize(self) -> super::isizex2::isizex2 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    fn to_usize(self) -> super::usizex2::usizex2 {
        unsafe { std::mem::transmute(self) }
    }
}
