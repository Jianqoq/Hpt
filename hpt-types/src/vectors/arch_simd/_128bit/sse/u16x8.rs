use crate::{
    traits::{SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2},
};

use std::arch::x86_64::*;

use crate::vectors::arch_simd::_128bit::i16x8;
use crate::vectors::arch_simd::_128bit::u16x8;

impl PartialEq for u16x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi16(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u16x8 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { u16x8(_mm_setzero_si128()) }
    }
}

impl VecTrait<u16> for u16x8 {
    const SIZE: usize = 8;
    type Base = u16;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u16x8(_mm_add_epi16(self.0, _mm_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: u16) -> u16x8 {
        unsafe { u16x8(_mm_set1_epi16(val as i16)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const u16) -> Self {
        unsafe { u16x8(_mm_loadu_si128(ptr as *const __m128i)) }
    }
}

impl SimdSelect<u16x8> for u16x8 {
    #[inline(always)]
    fn select(&self, true_val: u16x8, false_val: u16x8) -> u16x8 {
        unsafe { u16x8(_mm_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_sub_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            u16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Rem for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            u16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}

impl std::ops::BitAnd for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_and_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_or_si128(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(_mm_xor_si128(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { u16x8(_mm_xor_si128(self.0, _mm_set1_epi16(-1))) }
    }
}
impl std::ops::Shl for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 8] = std::mem::transmute(self.0);
            let b: [u16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}
impl std::ops::Shr for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 8] = std::mem::transmute(self.0);
            let b: [u16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            u16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
}

impl SimdMath<u16> for u16x8 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { u16x8(_mm_max_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { u16x8(_mm_min_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    fn relu(self) -> Self {
        self.max(Self::splat(0))
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        self.min(Self::splat(6)).max(Self::splat(0))
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        self
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        self
    }
    #[inline(always)]
    fn round(self) -> Self {
        self
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self
    }
    #[inline(always)]
    fn pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [u16; 8] = std::mem::transmute(self.0);
            let b: [u16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0u16; 8];
            for i in 0..8 {
                result[i] = a[i].pow(b[i] as u32);
            }
            u16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl FloatOutBinary2 for u16x8 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for u16")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for u16x8");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            u16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i))
        }
    }
}

impl Eval2 for u16x8 {
    type Output = i16x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i16x8::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let eq = _mm_cmpeq_epi16(self.0, _mm_setzero_si128());
            i16x8(_mm_xor_si128(eq, _mm_set1_epi16(-1)))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i16x8::default()
    }
}
