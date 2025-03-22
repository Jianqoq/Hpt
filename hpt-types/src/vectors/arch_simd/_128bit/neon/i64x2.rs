use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2},
};

use std::arch::aarch64::*;

use crate::vectors::arch_simd::_128bit::f64x2;
use crate::vectors::arch_simd::_128bit::i64x2;
#[cfg(target_pointer_width = "64")]
use crate::vectors::arch_simd::_128bit::isizex2;
use crate::vectors::arch_simd::_128bit::u64x2;
#[cfg(target_pointer_width = "64")]
use crate::vectors::arch_simd::_128bit::usizex2;

impl PartialEq for i64x2 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = vceqq_s64(self.0, other.0);
            vgetq_lane_u64(cmp, 0) == 0xffffffffffffffff
                && vgetq_lane_u64(cmp, 1) == 0xffffffffffffffff
        }
    }
}

impl Default for i64x2 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i64x2(vdupq_n_s64(0)) }
    }
}

impl VecTrait<i64> for i64x2 {
    const SIZE: usize = 2;
    type Base = i64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i64]) {
        unsafe {
            self.0 = vld1q_s64(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(a.0);
            let arr3: [i64; 2] = std::mem::transmute(b.0);
            let mut arr4: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            return i64x2(vld1q_s64(arr4.as_ptr()));
        }
    }
    #[inline(always)]
    fn sum(&self) -> i64 {
        unsafe { vaddvq_s64(self.0) }
    }
    #[inline(always)]
    fn splat(val: i64) -> i64x2 {
        unsafe { i64x2(vdupq_n_s64(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i64) -> Self {
        unsafe { i64x2(vld1q_s64(ptr)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
}

impl SimdCompare for i64x2 {
    type SimdMask = i64x2;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i64x2 {
        unsafe { i64x2(vreinterpretq_s64_u64(vceqq_s64(self.0, other.0))) }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i64x2 {
        unsafe {
            let eq = vceqq_s64(self.0, other.0);
            i64x2(veorq_s64(vreinterpretq_s64_u64(eq), vdupq_n_s64(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i64x2 {
        unsafe { i64x2(vreinterpretq_s64_u64(vcltq_s64(self.0, other.0))) }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i64x2 {
        unsafe { i64x2(vreinterpretq_s64_u64(vcleq_s64(self.0, other.0))) }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i64x2 {
        unsafe { i64x2(vreinterpretq_s64_u64(vcgtq_s64(self.0, other.0))) }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i64x2 {
        unsafe { i64x2(vreinterpretq_s64_u64(vcgeq_s64(self.0, other.0))) }
    }
}

impl SimdSelect<i64x2> for crate::vectors::arch_simd::_128bit::i64x2 {
    #[inline(always)]
    fn select(&self, true_val: i64x2, false_val: i64x2) -> i64x2 {
        unsafe {
            i64x2(vbslq_s64(
                vreinterpretq_u64_s64(self.0),
                true_val.0,
                false_val.0,
            ))
        }
    }
}

impl std::ops::Add for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(vaddq_s64(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(vsubq_s64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i].wrapping_mul(arr2[i]);
            }
            return i64x2(vld1q_s64(arr3.as_ptr()));
        }
    }
}
impl std::ops::Div for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            return i64x2(vld1q_s64(arr3.as_ptr()));
        }
    }
}
impl std::ops::Rem for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i] % arr2[i];
            }
            return i64x2(vld1q_s64(arr3.as_ptr()));
        }
    }
}
impl std::ops::Neg for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i64x2(vnegq_s64(self.0)) }
    }
}

impl std::ops::BitAnd for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(vandq_s64(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(vorrq_s64(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(veorq_s64(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i64x2(veorq_s64(self.0, vdupq_n_s64(-1))) }
    }
}
impl std::ops::Shl for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe { i64x2(vshlq_s64(self.0, rhs.0)) }
    }
}
impl std::ops::Shr for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i64; 2] = std::mem::transmute(self.0);
            let b: [i64; 2] = std::mem::transmute(rhs.0);
            let mut result = [0; 2];
            for i in 0..2 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            return i64x2(vld1q_s64(result.as_ptr()));
        }
    }
}
impl SimdMath<i64> for i64x2 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(other.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i].max(arr2[i]);
            }
            return i64x2(vld1q_s64(arr3.as_ptr()));
        }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(other.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                arr3[i] = arr[i].min(arr2[i]);
            }
            return i64x2(vld1q_s64(arr3.as_ptr()));
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
        unsafe { i64x2(vabsq_s64(self.0)) }
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
            let a: [i64; 2] = std::mem::transmute(self.0);
            let b: [i64; 2] = std::mem::transmute(rhs.0);
            let mut result = [0i64; 2];
            for i in 0..2 {
                result[i] = a[i].pow(b[i] as u32);
            }
            return i64x2(vld1q_s64(result.as_ptr()));
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(i64x2::splat(0)) + alpha * self.min(i64x2::splat(0))
    }
}

impl VecConvertor for i64x2 {
    #[inline(always)]
    fn to_i64(self) -> i64x2 {
        self
    }
    #[inline(always)]
    fn to_u64(self) -> u64x2 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f64(self) -> f64x2 {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let mut result = [0.0f64; 2];
            for i in 0..2 {
                result[i] = arr[i] as f64;
            }
            return f64x2(vld1q_f64(result.as_ptr()));
        }
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_isize(self) -> isizex2 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_usize(self) -> usizex2 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for i64x2 {
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
        panic!("Hypot operation is not supported for i64x2");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [i64; 2] = std::mem::transmute(self.0);
            let arr2: [i64; 2] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 2] = [0; 2];
            for i in 0..2 {
                if arr2[i] < 0 {
                    panic!("Power operation is not supported for negative i64");
                }
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            return i64x2(vld1q_s64(arr3.as_ptr()));
        }
    }
}

impl Eval2 for i64x2 {
    type Output = i64x2;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i64x2::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let eq = vceqq_s64(self.0, vdupq_n_s64(0));
            let one = vdupq_n_s64(1);
            i64x2(vbicq_s64(one, vreinterpretq_s64_u64(eq)))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i64x2::default()
    }
}
