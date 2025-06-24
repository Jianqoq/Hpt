use crate::{
    convertion::VecConvertor,
    traits::{SimdMath, SimdSelect, VecTrait},
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

impl PartialEq for u64x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let mask = _mm512_cmpeq_epu64_mask(self.0, other.0);
            mask == 0xff
        }
    }
}

impl Default for u64x8 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { u64x8(_mm512_setzero_si512()) }
    }
}

impl VecTrait<u64> for u64x8 {
    const SIZE: usize = 8;
    type Base = u64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [u64; 8] = std::mem::transmute(self.0);
            let arr2: [u64; 8] = std::mem::transmute(a.0);
            let arr3: [u64; 8] = std::mem::transmute(b.0);
            let mut arr4: [u64; 8] = [0; 8];
            for i in 0..8 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            u64x8(_mm512_loadu_si512(arr4.as_ptr() as *const __m512i))
        }
    }
    #[inline(always)]
    fn sum(&self) -> u64 {
        unsafe {
            let arr: [u64; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: u64) -> u64x8 {
        unsafe { u64x8(_mm512_set1_epi64(val as i64)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const u64) -> Self {
        u64x8(_mm512_loadu_si512(ptr as *const __m512i))
    }
}

impl SimdSelect<u64x8> for i64x8 {
    #[inline(always)]
    fn select(&self, true_val: u64x8, false_val: u64x8) -> u64x8 {
        unsafe {
            let mask = _mm512_movepi64_mask(self.0);
            u64x8(_mm512_mask_blend_epi64(mask, false_val.0, true_val.0))
        }
    }
}

impl std::ops::Add for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { u64x8(_mm512_add_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { u64x8(_mm512_sub_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 8] = std::mem::transmute(self.0);
            let arr2: [u64; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] * arr2[i];
            }
            u64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Div for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 8] = std::mem::transmute(self.0);
            let arr2: [u64; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 8] = [0; 8];
            for i in 0..8 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            u64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Rem for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 8] = std::mem::transmute(self.0);
            let arr2: [u64; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            u64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::BitAnd for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { u64x8(_mm512_and_si512(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { u64x8(_mm512_or_si512(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { u64x8(_mm512_xor_si512(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { u64x8(_mm512_xor_si512(self.0, _mm512_set1_epi64(-1))) }
    }
}
impl std::ops::Shl for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        unsafe {
            let a: [u64; 8] = std::mem::transmute(self.0);
            let b: [u64; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u64x8(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Shr for u64x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        unsafe {
            let a: [u64; 8] = std::mem::transmute(self.0);
            let b: [u64; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            u64x8(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
}
impl SimdMath<u64> for u64x8 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
            let arr: [u64; 8] = std::mem::transmute(self.0);
            let arr2: [u64; 8] = std::mem::transmute(other.0);
            let mut arr3: [u64; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i].max(arr2[i]);
            }
            u64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            let arr: [u64; 8] = std::mem::transmute(self.0);
            let arr2: [u64; 8] = std::mem::transmute(other.0);
            let mut arr3: [u64; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i].min(arr2[i]);
            }
            u64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
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
        self
    }
    #[inline(always)]
    fn pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [u64; 8] = std::mem::transmute(self.0);
            let b: [u64; 8] = std::mem::transmute(rhs.0);
            let mut result = [0u64; 8];
            for i in 0..8 {
                result[i] = a[i].pow(b[i] as u32);
            }
            u64x8(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl VecConvertor for u64x8 {
    #[inline(always)]
    fn to_u64(self) -> u64x8 {
        self
    }
    #[inline(always)]
    fn to_i64(self) -> i64x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f64(self) -> f64x8 {
        unsafe {
            let arr: [u64; 8] = std::mem::transmute(self.0);
            let mut result = [0.0f64; 8];
            for i in 0..8 {
                result[i] = arr[i] as f64;
            }
            f64x8(_mm512_loadu_pd(result.as_ptr()))
        }
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_isize(self) -> isizex8 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_usize(self) -> usizex8 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for u64x8 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for u64")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for u64x8");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 8] = std::mem::transmute(self.0);
            let arr2: [u64; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            u64x8(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}

impl Eval2 for u64x8 {
    type Output = i64x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i64x8::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let mask = _mm512_cmpneq_epi64_mask(self.0, _mm512_setzero_si512());
            i64x8(_mm512_maskz_mov_epi64(mask, _mm512_set1_epi64(1)))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i64x8::default()
    }
}
