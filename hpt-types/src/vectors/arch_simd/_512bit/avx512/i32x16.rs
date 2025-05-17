use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOutUnary2},
};
use std::arch::x86_64::*;

use crate::simd::_512bit::f32x16;
use crate::simd::_512bit::i32x16;
use crate::simd::_512bit::u32x16;

impl PartialEq for i32x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let mask = _mm512_cmpeq_epi32_mask(self.0, other.0);
            mask == 0xffff
        }
    }
}

impl Default for i32x16 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i32x16(_mm512_setzero_si512()) }
    }
}

impl VecTrait<i32> for i32x16 {
    const SIZE: usize = 8;
    type Base = i32;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i32x16(_mm512_add_epi32(self.0, _mm512_mullo_epi32(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> i32 {
        unsafe {
            let arr: [i32; 16] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: i32) -> i32x16 {
        unsafe { i32x16(_mm512_set1_epi32(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i32) -> Self {
        i32x16(_mm512_loadu_si512(ptr as *const __m512i))
    }
}

impl SimdCompare for i32x16 {
    type SimdMask = i32x16;

    #[inline(always)]
    fn simd_eq(self, other: Self) -> i32x16 {
        unsafe {
            let mask = _mm512_cmpeq_epi32_mask(self.0, other.0);
            i32x16(_mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1)))
        }
    }

    #[inline(always)]
    fn simd_ne(self, other: Self) -> i32x16 {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(self.0, other.0);
            i32x16(_mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1)))
        }
    }

    #[inline(always)]
    fn simd_lt(self, other: Self) -> i32x16 {
        unsafe {
            let mask = _mm512_cmplt_epi32_mask(self.0, other.0);
            i32x16(_mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1)))
        }
    }

    #[inline(always)]
    fn simd_le(self, other: Self) -> i32x16 {
        unsafe {
            let mask = _mm512_cmple_epi32_mask(self.0, other.0);
            i32x16(_mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1)))
        }
    }

    #[inline(always)]
    fn simd_gt(self, other: Self) -> i32x16 {
        unsafe {
            let mask = _mm512_cmpgt_epi32_mask(self.0, other.0);
            i32x16(_mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1)))
        }
    }

    #[inline(always)]
    fn simd_ge(self, other: Self) -> i32x16 {
        unsafe {
            let mask = _mm512_cmpge_epi32_mask(self.0, other.0);
            i32x16(_mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1)))
        }
    }
}

impl SimdSelect<i32x16> for i32x16 {
    #[inline(always)]
    fn select(&self, true_val: i32x16, false_val: i32x16) -> i32x16 {
        unsafe {
            let mask = _mm512_movepi32_mask(self.0);
            i32x16(_mm512_mask_blend_epi32(mask, false_val.0, true_val.0))
        }
    }
}

impl std::ops::Add for i32x16 {
    type Output = i32x16;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i32x16(_mm512_add_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i32x16 {
    type Output = i32x16;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i32x16(_mm512_sub_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i32x16 {
    type Output = i32x16;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i32x16(_mm512_mullo_epi32(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i32x16 {
    type Output = i32x16;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 16] = std::mem::transmute(self.0);
            let arr2: [i32; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 16] = [0; 16];
            for i in 0..16 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            i32x16(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Rem for i32x16 {
    type Output = i32x16;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 16] = std::mem::transmute(self.0);
            let arr2: [i32; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] % arr2[i];
            }
            i32x16(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Neg for i32x16 {
    type Output = i32x16;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i32x16(_mm512_sub_epi32(_mm512_setzero_si512(), self.0)) }
    }
}
impl std::ops::BitAnd for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i32x16(_mm512_and_si512(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i32x16(_mm512_or_si512(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i32x16(_mm512_xor_si512(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i32x16(_mm512_xor_si512(self.0, _mm512_set1_epi32(-1))) }
    }
}
impl std::ops::Shl for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 16] = std::mem::transmute(self.0);
            let b: [i32; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            i32x16(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
}
impl std::ops::Shr for i32x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 16] = std::mem::transmute(self.0);
            let b: [i32; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i32x16(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
}
impl SimdMath<i32> for i32x16 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { i32x16(_mm512_max_epi32(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { i32x16(_mm512_min_epi32(self.0, other.0)) }
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
        unsafe { i32x16(_mm512_abs_epi32(self.0)) }
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
            let a: [i32; 16] = std::mem::transmute(self.0);
            let b: [i32; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i32; 16];
            for i in 0..16 {
                result[i] = a[i].pow(b[i] as u32);
            }
            i32x16(_mm512_loadu_si512(result.as_ptr() as *const __m512i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl VecConvertor for i32x16 {
    #[inline(always)]
    fn to_i32(self) -> i32x16 {
        self
    }
    #[inline(always)]
    fn to_u32(self) -> u32x16 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f32(self) -> f32x16 {
        unsafe { f32x16(_mm512_cvtepi32_ps(self.0)) }
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

impl FloatOutBinary2 for i32x16 {
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
        panic!("Hypot operation is not supported for i32x16");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [i32; 16] = std::mem::transmute(self.0);
            let arr2: [i32; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 16] = [0; 16];
            for i in 0..16 {
                if arr2[i] < 0 {
                    panic!("Power operation is not supported for negative i32");
                }
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            i32x16(_mm512_loadu_si512(arr3.as_ptr() as *const __m512i))
        }
    }
}

impl NormalOutUnary2 for i32x16 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        i32x16(unsafe { _mm512_abs_epi32(self.0) })
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
        unsafe { Self(_mm512_sub_epi32(_mm512_setzero_si512(), self.0)) }
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

impl Eval2 for i32x16 {
    type Output = i32x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i32x16::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let mask = _mm512_cmpneq_epi32_mask(self.0, _mm512_setzero_si512());
            Self(_mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(1)))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i32x16::default()
    }
}
