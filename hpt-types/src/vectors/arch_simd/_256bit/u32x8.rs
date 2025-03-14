use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i32x8::i32x8;

/// a vector of 4 u32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u32x8(pub(crate) __m256i);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type u32_promote = u32x8;

impl Default for u32x8 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { u32x8(_mm256_setzero_si256()) }
    }
}

impl PartialEq for u32x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi32(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}
impl VecTrait<u32> for u32x8 {
    const SIZE: usize = 8;
    type Base = u32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u32]) {
        unsafe {
            _mm256_storeu_si256(
                &mut self.0,
                _mm256_loadu_si256(slice.as_ptr() as *const __m256i),
            )
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { u32x8(_mm256_add_epi32(self.0, _mm256_mullo_epi32(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> u32 {
        unsafe {
            let arr: [u32; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: u32) -> u32x8 {
        unsafe { u32x8(_mm256_set1_epi32(val as i32)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const u32) -> Self {
        u32x8(_mm256_loadu_si256(ptr as *const __m256i))
    }
}

impl u32x8 {
    /// convert the vector to an array
    #[allow(unused)]
    #[inline(always)]
    pub fn as_array(&self) -> [u32; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for u32x8 {
    type SimdMask = i32x8;

    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }

    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }

    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }

    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }

    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }

    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl SimdSelect<u32x8> for i32x8 {
    #[inline(always)]
    fn select(&self, true_val: u32x8, false_val: u32x8) -> u32x8 {
        unsafe { u32x8(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for u32x8 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_add_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u32x8 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_sub_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_mullo_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u32; 8] = std::mem::transmute(self.0);
            let arr2: [u32; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u32; 8] = [0; 8];
            for i in 0..8 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            u32x8(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u32; 8] = std::mem::transmute(self.0);
            let arr2: [u32; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u32; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            u32x8(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::BitAnd for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { u32x8(_mm256_xor_si256(self.0, _mm256_set1_epi32(-1))) }
    }
}
impl std::ops::Shl for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u32; 8] = std::mem::transmute(self.0);
            let b: [u32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for u32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u32; 8] = std::mem::transmute(self.0);
            let b: [u32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            u32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}

impl SimdMath<u32> for u32x8 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { u32x8(_mm256_max_epu32(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { u32x8(_mm256_min_epu32(self.0, other.0)) }
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
            let a: [u32; 8] = std::mem::transmute(self.0);
            let b: [u32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0u32; 8];
            for i in 0..8 {
                result[i] = a[i].pow(b[i] as u32);
            }
            u32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl VecConvertor for u32x8 {
    #[inline(always)]
    fn to_u32(self) -> u32x8 {
        self
    }
    #[inline(always)]
    fn to_i32(self) -> i32x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f32(self) -> super::f32x8::f32x8 {
        unsafe {
            let arr: [u32; 8] = std::mem::transmute(self.0);
            let mut result = [0.0f32; 8];
            for i in 0..8 {
                result[i] = arr[i] as f32;
            }
            super::f32x8::f32x8(_mm256_loadu_ps(result.as_ptr()))
        }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_usize(self) -> super::usizex4::usizex4 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_isize(self) -> super::isizex4::isizex4 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for u32x8 {
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
        panic!("Hypot operation is not supported for u32x8");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u32; 8] = std::mem::transmute(self.0);
            let arr2: [u32; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u32; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i].pow(arr2[i]);
            }
            u32x8(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}

impl NormalOut2 for u32x8 {
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

impl NormalOutUnary2 for u32x8 {
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

impl Eval2 for u32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i32x8::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let eq = _mm256_cmpeq_epi32(self.0, _mm256_setzero_si256());
            let result = _mm256_andnot_si256(eq, _mm256_set1_epi32(1));
            i32x8(result)
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i32x8::default()
    }
}
