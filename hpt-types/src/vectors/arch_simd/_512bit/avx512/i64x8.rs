use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2},
};

use std::arch::x86_64::*;

use crate::simd::_512bit::f64x8;
use crate::simd::_512bit::i64x8;
#[cfg(target_pointer_width = "64")]
use crate::simd::_512bit::isizex8;
use crate::simd::_512bit::u64x8;
#[cfg(target_pointer_width = "64")]
use crate::simd::_512bit::usizex8;

impl PartialEq for i64x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let mask = _mm512_cmpeq_epi64_mask(self.0, other.0);
            mask == 0xff
        }
    }
}

impl Default for i64x8 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i64x8(_mm512_setzero_si512()) }
    }
}

impl VecTrait<i64> for i64x8 {
    const SIZE: usize = 8;
    type Base = i64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            let arr2: [i64; 8] = std::mem::transmute(a.0);
            let arr3: [i64; 8] = std::mem::transmute(b.0);
            let mut arr4: [i64; 8] = [0; 8];
            for i in 0..8 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            i64x8(_mm512_loadu_si512(arr4.as_ptr() as *const __m512i))
        }
    }
    #[inline(always)]
    fn sum(&self) -> i64 {
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: i64) -> i64x8 {
        unsafe { i64x8(_mm512_set1_epi64(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i64) -> Self {
        i64x8(_mm512_loadu_si512(ptr as *const __m512i))
    }
}

impl SimdCompare for i64x8 {
    type SimdMask = i64x8;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i64x8 {
        unsafe {
            let mask = _mm512_cmpeq_epi64_mask(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i64x8 {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i64x8 {
        unsafe {
            let mask = _mm512_cmplt_epi64_mask(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i64x8 {
        unsafe {
            let mask = _mm512_cmple_epi64_mask(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i64x8 {
        unsafe {
            let mask = _mm512_cmpgt_epi64_mask(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i64x8 {
        unsafe {
            let mask = _mm512_cmpge_epi64_mask(self.0, other.0);
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(-1)))
        }
    }
}

impl SimdSelect<i64x8> for i64x8 {
    #[inline(always)]
    fn select(&self, true_val: i64x8, false_val: i64x8) -> i64x8 {
        unsafe {
            let mask = _mm512_movepi64_mask(self.0);
            i64x8(_mm512_mask_blend_epi64(mask, false_val.0, true_val.0))
        }
    }
}

impl std::ops::Add for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i64x8(_mm512_add_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i64x8(_mm512_sub_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            let arr2: [i64; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] * arr2[i];
            }
            i64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Div for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            let arr2: [i64; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 8] = [0; 8];
            for i in 0..8 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            i64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Rem for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            let arr2: [i64; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            i64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Neg for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i64x8(_mm512_sub_epi64(_mm512_setzero_si512(), self.0)) }
    }
}

impl std::ops::BitAnd for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i64x8(_mm512_and_si512(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i64x8(_mm512_or_si512(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i64x8(_mm512_xor_si512(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i64x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i64x8(_mm512_xor_si512(self.0, _mm512_set1_epi64(-1))) }
    }
}
impl std::ops::Shl for i64x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i64; 8] = std::mem::transmute(self.0);
            let b: [i64; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            i64x8(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Shr for i64x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i64; 8] = std::mem::transmute(self.0);
            let b: [i64; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i64x8(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
}
impl SimdMath<i64> for i64x8 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            let arr2: [i64; 8] = std::mem::transmute(other.0);
            let mut arr3: [i64; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i].max(arr2[i]);
            }
            i64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            let arr2: [i64; 8] = std::mem::transmute(other.0);
            let mut arr3: [i64; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i].min(arr2[i]);
            }
            i64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
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
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            let mut result = [0i64; 8];
            for i in 0..8 {
                result[i] = arr[i].abs();
            }
            i64x8(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
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
            let a: [i64; 8] = std::mem::transmute(self.0);
            let b: [i64; 8] = std::mem::transmute(rhs.0);
            let mut result = [0i64; 8];
            for i in 0..8 {
                result[i] = a[i].pow(b[i] as u32);
            }
            i64x8(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl VecConvertor for i64x8 {
    #[inline(always)]
    fn to_i64(self) -> i64x8 {
        self
    }
    #[inline(always)]
    fn to_u64(self) -> u64x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f64(self) -> f64x8 {
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            let mut result = [0.0f64; 8];
            for i in 0..8 {
                result[i] = arr[i] as f64;
            }
            f64x8(_mm512_loadu_pd(result.as_ptr()))
        }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "64")]
    fn to_isize(self) -> isizex8 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "64")]
    fn to_usize(self) -> usizex8 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for i64x8 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i32")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for i64x8");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [i64; 8] = std::mem::transmute(self.0);
            let arr2: [i64; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 8] = [0; 8];
            for i in 0..8 {
                if arr2[i] < 0 {
                    panic!("Power operation is not supported for negative i64");
                }
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            i64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}

impl Eval2 for i64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i64x8::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(self.0, _mm512_setzero_si512());
            Self(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(1)))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i64x8::default()
    }
}
