use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::u32x8::u32x8;

/// a vector of 8 i32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct i32x8(pub(crate) __m256i);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type i32_promote = i32x8;

impl PartialEq for i32x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi32(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i32x8 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i32x8(_mm256_setzero_si256()) }
    }
}

impl VecTrait<i32> for i32x8 {
    const SIZE: usize = 8;
    type Base = i32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i32]) {
        unsafe {
            _mm256_storeu_si256(
                &mut self.0,
                _mm256_loadu_si256(slice.as_ptr() as *const __m256i),
            )
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i32x8(_mm256_add_epi32(self.0, _mm256_mullo_epi32(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> i32 {
        unsafe {
            let arr: [i32; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: i32) -> i32x8 {
        unsafe { i32x8(_mm256_set1_epi32(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i32) -> Self {
        i32x8(_mm256_loadu_si256(ptr as *const __m256i))
    }
}

impl i32x8 {
    /// convert the vector to an array
    #[inline(always)]
    pub fn as_array(&self) -> [i32; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for i32x8 {
    type SimdMask = i32x8;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i32x8 {
        unsafe { i32x8(_mm256_cmpeq_epi32(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i32x8 {
        unsafe {
            let eq = _mm256_cmpeq_epi32(self.0, other.0);
            i32x8(_mm256_xor_si256(eq, _mm256_set1_epi32(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i32x8 {
        unsafe { i32x8(_mm256_cmpgt_epi32(other.0, self.0)) }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i32x8 {
        unsafe {
            let lt = _mm256_cmpgt_epi32(other.0, self.0);
            let eq = _mm256_cmpeq_epi32(self.0, other.0);
            i32x8(_mm256_or_si256(lt, eq))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i32x8 {
        unsafe { i32x8(_mm256_cmpgt_epi32(self.0, other.0)) }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i32x8 {
        unsafe {
            let gt = _mm256_cmpgt_epi32(self.0, other.0);
            let eq = _mm256_cmpeq_epi32(self.0, other.0);
            i32x8(_mm256_or_si256(gt, eq))
        }
    }
}

impl SimdSelect<i32x8> for crate::vectors::arch_simd::_256bit::i32x8::i32x8 {
    #[inline(always)]
    fn select(&self, true_val: i32x8, false_val: i32x8) -> i32x8 {
        unsafe { i32x8(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_add_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_sub_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_mullo_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 8] = std::mem::transmute(self.0);
            let arr2: [i32; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 8] = [0; 8];
            for i in 0..8 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            i32x8(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 8] = std::mem::transmute(self.0);
            let arr2: [i32; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            i32x8(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Neg for i32x8 {
    type Output = i32x8;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i32x8(_mm256_sign_epi32(self.0, _mm256_set1_epi32(-1))) }
    }
}
impl std::ops::BitAnd for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i32x8(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i32x8(_mm256_xor_si256(self.0, _mm256_set1_epi32(-1))) }
    }
}
impl std::ops::Shl for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 8] = std::mem::transmute(self.0);
            let b: [i32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            i32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 8] = std::mem::transmute(self.0);
            let b: [i32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl SimdMath<i32> for i32x8 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { i32x8(_mm256_max_epi32(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { i32x8(_mm256_min_epi32(self.0, other.0)) }
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
    fn square(self) -> Self {
        self * self
    }
    #[inline(always)]
    fn abs(self) -> Self {
        unsafe { i32x8(_mm256_abs_epi32(self.0)) }
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }
    #[inline(always)]
    fn signum(self) -> Self {
        let zero = Self::splat(0);
        let gt = self.simd_gt(zero);
        let lt = self.simd_lt(zero);
        let pos = gt & Self::splat(1);
        let neg = lt & Self::splat(-1);
        pos | neg
    }
    #[inline(always)]
    fn pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [i32; 8] = std::mem::transmute(self.0);
            let b: [i32; 8] = std::mem::transmute(rhs.0);
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = a[i].pow(b[i] as u32);
            }
            i32x8(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl VecConvertor for i32x8 {
    #[inline(always)]
    fn to_i32(self) -> i32x8 {
        self
    }
    #[inline(always)]
    fn to_u32(self) -> u32x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f32(self) -> super::f32x8::f32x8 {
        unsafe { super::f32x8::f32x8(_mm256_cvtepi32_ps(self.0)) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_isize(self) -> super::isizex4::isizex4 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_usize(self) -> super::usizex4::usizex4 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for i32x8 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i32")
    }
}

impl NormalOut2 for i32x8 {
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

impl NormalOutUnary2 for i32x8 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        i32x8(unsafe { _mm256_abs_epi32(self.0) })
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
        unsafe { Self(_mm256_sub_epi32(_mm256_setzero_si256(), self.0)) }
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
        self.max(i32x8::splat(0)) + alpha * self.min(i32x8::splat(0))
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
}

impl Eval2 for i32x8 {
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
            Self(result)
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i32x8::default()
    }
}
