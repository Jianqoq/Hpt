use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 8 i16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct i16x8(pub(crate) __m128i);

impl PartialEq for i16x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi16(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}
impl Default for i16x8 {
    fn default() -> Self {
        unsafe { i16x8(_mm_setzero_si128()) }
    }
}
impl VecTrait<i16> for i16x8 {
    const SIZE: usize = 8;
    type Base = i16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i16]) {
        unsafe {
            _mm_storeu_si128(
                &mut self.0,
                _mm_loadu_si128(slice.as_ptr() as *const __m128i),
            )
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i16x8(_mm_add_epi16(self.0, _mm_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> i16 {
        unsafe {
            let arr: [i16; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    fn splat(val: i16) -> i16x8 {
        unsafe { i16x8(_mm_set1_epi16(val)) }
    }
}

impl i16x8 {
    #[allow(unused)]
    fn as_array(&self) -> [i16; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for i16x8 {
    type SimdMask = i16x8;
    fn simd_eq(self, other: Self) -> i16x8 {
        unsafe { i16x8(_mm_cmpeq_epi16(self.0, other.0)) }
    }
    fn simd_ne(self, other: Self) -> i16x8 {
        unsafe { 
            let eq = _mm_cmpeq_epi16(self.0, other.0);
            i16x8(_mm_xor_si128(eq, _mm_set1_epi16(-1)))
        }
    }
    fn simd_lt(self, other: Self) -> i16x8 {
        unsafe { i16x8(_mm_cmplt_epi16(self.0, other.0)) }
    }
    fn simd_le(self, other: Self) -> i16x8 {
        unsafe { 
            let lt = _mm_cmplt_epi16(self.0, other.0);
            let eq = _mm_cmpeq_epi16(self.0, other.0);
            i16x8(_mm_or_si128(lt, eq))
        }
    }
    fn simd_gt(self, other: Self) -> i16x8 {
        unsafe { i16x8(_mm_cmpgt_epi16(self.0, other.0)) }
    }
    fn simd_ge(self, other: Self) -> i16x8 {
        unsafe { 
            let gt = _mm_cmpgt_epi16(self.0, other.0);
            let eq = _mm_cmpeq_epi16(self.0, other.0);
            i16x8(_mm_or_si128(gt, eq))
        }
    }
}

impl SimdSelect<i16x8> for crate::vectors::arch_simd::_128bit::i16x8::i16x8 {
    fn select(&self, true_val: i16x8, false_val: i16x8) -> i16x8 {
        unsafe { i16x8(_mm_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for i16x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i16x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_sub_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i16x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i16x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i16; 8] = std::mem::transmute(self.0);
            let arr2: [i16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] / arr2[i];
            }
            i16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for i16x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i16; 8] = std::mem::transmute(self.0);
            let arr2: [i16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            i16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Neg for i16x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe { i16x8(_mm_sub_epi16(_mm_setzero_si128(), self.0)) }
    }
}
impl std::ops::BitAnd for i16x8 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_and_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i16x8 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_or_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i16x8 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i16x8(_mm_xor_si128(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i16x8 {
    type Output = Self;
    fn not(self) -> Self::Output {
        unsafe { i16x8(_mm_xor_si128(self.0, _mm_set1_epi16(-1))) }
    }
}
impl std::ops::Shl for i16x8 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i16; 8] = std::mem::transmute(self.0);
            let b: [i16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i] << b[i];
            }
            i16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Shr for i16x8 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i16; 8] = std::mem::transmute(self.0);
            let b: [i16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i] >> b[i];
            }
            i16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}

impl SimdMath<i16> for i16x8 {
    fn max(self, other: Self) -> Self {
        unsafe { i16x8(_mm_max_epi16(self.0, other.0)) }
    }
    fn min(self, other: Self) -> Self {
        unsafe { i16x8(_mm_min_epi16(self.0, other.0)) }
    }
    fn relu(self) -> Self {
        unsafe { i16x8(_mm_max_epi16(self.0, _mm_setzero_si128())) }
    }
    fn relu6(self) -> Self {
        unsafe { i16x8(_mm_min_epi16(self.relu().0, _mm_set1_epi16(6))) }
    }
}

impl VecConvertor for i16x8 {}
