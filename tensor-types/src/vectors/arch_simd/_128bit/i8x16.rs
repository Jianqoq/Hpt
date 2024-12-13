use crate::{convertion::VecConvertor, traits::{SimdCompare, SimdMath, VecTrait}};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 16 i8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct i8x16(pub(crate) __m128i);

impl PartialEq for i8x16 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi8(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i8x16 {
    fn default() -> Self {
        unsafe { i8x16(_mm_setzero_si128()) }
    }
}

impl VecTrait<i8> for i8x16 {
    const SIZE: usize = 16;
    type Base = i8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i8]) {
        unsafe {
            _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i));
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i8x16(_mm_add_epi8(_mm_mullo_epi16(self.0, a.0), b.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        unsafe {
            let sum = _mm_sad_epu8(self.0, _mm_setzero_si128());
            _mm_cvtsi128_si32(sum) as i8
        }
    }
    fn splat(val: i8) -> i8x16 {
        unsafe { i8x16(_mm_set1_epi8(val)) }
    }
}

impl i8x16 {
    #[allow(unused)]
    fn as_array(&self) -> [i8; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for i8x16 {
    type SimdMask = i8x16;
    fn simd_eq(self, other: Self) -> i8x16 {
        unsafe { i8x16(_mm_cmpeq_epi8(self.0, other.0)) }
    }
    fn simd_ne(self, other: Self) -> i8x16 {
        unsafe { 
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            i8x16(_mm_xor_si128(eq, _mm_set1_epi8(-1)))
        }
    }
    fn simd_lt(self, other: Self) -> i8x16 {
        unsafe { i8x16(_mm_cmplt_epi8(self.0, other.0)) }
    }
    fn simd_le(self, other: Self) -> i8x16 {
        unsafe { 
            let lt = _mm_cmplt_epi8(self.0, other.0);
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            i8x16(_mm_or_si128(lt, eq))
        }
    }
    fn simd_gt(self, other: Self) -> i8x16 {
        unsafe { i8x16(_mm_cmpgt_epi8(self.0, other.0)) }
    }
    fn simd_ge(self, other: Self) -> i8x16 {
        unsafe { 
            let gt = _mm_cmpgt_epi8(self.0, other.0);
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            i8x16(_mm_or_si128(gt, eq))
        }
    }
}

impl std::ops::Add for i8x16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_add_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i8x16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_sub_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i8x16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i8x16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] / b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for i8x16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] % b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Neg for i8x16 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        unsafe { i8x16(_mm_sub_epi8(_mm_setzero_si128(), self.0)) }
    }
}
impl std::ops::BitAnd for i8x16 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_and_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i8x16 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_or_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i8x16 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(_mm_xor_si128(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i8x16 {
    type Output = Self;
    fn not(self) -> Self::Output {
        unsafe { i8x16(_mm_xor_si128(self.0, _mm_set1_epi8(-1))) }
    }
}
impl std::ops::Shl for i8x16 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] << b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Shr for i8x16 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] >> b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}

impl SimdMath<i8> for i8x16 {
    fn max(self, other: Self) -> Self {
        unsafe { i8x16(_mm_max_epi8(self.0, other.0)) }
    }
    fn min(self, other: Self) -> Self {
        unsafe { i8x16(_mm_min_epi8(self.0, other.0)) }
    }
    fn relu(self) -> Self {
        unsafe { i8x16(_mm_max_epi8(self.0, _mm_setzero_si128())) }
    }
    fn relu6(self) -> Self {
        unsafe { i8x16(_mm_min_epi8(self.relu().0, _mm_set1_epi8(6))) }
    }
}

impl VecConvertor for i8x16 {
}
