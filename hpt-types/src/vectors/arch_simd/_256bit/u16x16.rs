use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i16x16::i16x16;

/// a vector of 16 u16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u16x16(pub(crate) __m256i);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type u16_promote = u16x16;

impl PartialEq for u16x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi16(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u16x16 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { u16x16(_mm256_setzero_si256()) }
    }
}

impl VecTrait<u16> for u16x16 {
    const SIZE: usize = 16;
    type Base = u16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u16]) {
        unsafe {
            _mm256_storeu_si256(
                &mut self.0,
                _mm256_loadu_si256(slice.as_ptr() as *const __m256i),
            )
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u16x16(_mm256_add_epi16(self.0, _mm256_mullo_epi16(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: u16) -> u16x16 {
        unsafe { u16x16(_mm256_set1_epi16(val as i16)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const u16) -> Self {
        u16x16(_mm256_loadu_si256(ptr as *const __m256i))
    }
}

impl u16x16 {
    /// convert the vector to an array
    #[allow(unused)]
    pub fn as_array(&self) -> [u16; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdSelect<u16x16> for u16x16 {
    #[inline(always)]
    fn select(&self, true_val: u16x16, false_val: u16x16) -> u16x16 {
        unsafe { u16x16(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_add_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_sub_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_mullo_epi16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            let arr2: [u16; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 16] = [0; 16];
            for i in 0..16 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            u16x16(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            let arr2: [u16; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] % arr2[i];
            }
            u16x16(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}

impl std::ops::BitAnd for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u16x16 {
    type Output = u16x16;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { u16x16(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u16x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { u16x16(_mm256_xor_si256(self.0, _mm256_set1_epi16(-1))) }
    }
}
impl std::ops::Shl for u16x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 16] = std::mem::transmute(self.0);
            let b: [u16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for u16x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 16] = std::mem::transmute(self.0);
            let b: [u16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            u16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl SimdCompare for u16x16 {
    type SimdMask = i16x16;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl SimdMath<u16> for u16x16 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { u16x16(_mm256_max_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { u16x16(_mm256_min_epu16(self.0, other.0)) }
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
            let a: [u16; 16] = std::mem::transmute(self.0);
            let b: [u16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0u16; 16];
            for i in 0..16 {
                result[i] = a[i].pow(b[i] as u32);
            }
            u16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl VecConvertor for u16x16 {
    #[inline(always)]
    fn to_u16(self) -> u16x16 {
        self
    }
    #[inline(always)]
    fn to_i16(self) -> i16x16 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f16(self) -> super::f16x16::f16x16 {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            let mut result = [half::f16::ZERO; 16];
            for i in 0..16 {
                result[i] = half::f16::from_f32(arr[i] as f32);
            }
            super::f16x16::f16x16(result)
        }
    }
    #[inline(always)]
    fn to_bf16(self) -> super::bf16x16::bf16x16 {
        unsafe {
            let arr: [u16; 16] = std::mem::transmute(self.0);
            let mut result = [half::bf16::ZERO; 16];
            for i in 0..16 {
                result[i] = half::bf16::from_f32(arr[i] as f32);
            }
            super::bf16x16::bf16x16(result)
        }
    }
}

impl FloatOutBinary2 for u16x16 {
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
        panic!("Hypot operation is not supported for u16x16");
    }
}

impl NormalOut2 for u16x16 {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn __sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        self.pow(rhs)
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl NormalOutUnary2 for u16x16 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        self
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        self.signum()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.leaky_relu(alpha)
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.relu()
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.relu6()
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        self
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        self.abs() * rhs.signum()
    }
}

impl Eval2 for u16x16 {
    type Output = i16x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i16x16::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let eq = _mm256_cmpeq_epi16(self.0, _mm256_setzero_si256());
            let result = _mm256_andnot_si256(eq, _mm256_set1_epi16(1));
            i16x16(result)
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i16x16::default()
    }
}
