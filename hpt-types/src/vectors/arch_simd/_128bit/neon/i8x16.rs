use crate::{
    traits::{SimdCompare, SimdMath, VecTrait},
    type_promote::{Eval2, FloatOutBinary2},
};
use std::arch::aarch64::*;

use crate::vectors::arch_simd::_128bit::i8x16;

impl PartialEq for i8x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = vceqq_s8(self.0, other.0);
            vmaxvq_u8(cmp) == 0xff && vminvq_u8(cmp) == 0xff
        }
    }
}

impl Default for i8x16 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { i8x16(vdupq_n_s8(0)) }
    }
}

impl VecTrait<i8> for i8x16 {
    const SIZE: usize = 16;
    type Base = i8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i8]) {
        unsafe {
            self.0 = vld1q_s8(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { i8x16(vmlaq_s8(self.0, a.0, b.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        unsafe { vaddvq_s8(self.0) as i8 }
    }
    #[inline(always)]
    fn splat(val: i8) -> i8x16 {
        unsafe { i8x16(vdupq_n_s8(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i8) -> Self {
        unsafe { i8x16(vld1q_s8(ptr)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
}

impl SimdCompare for i8x16 {
    type SimdMask = i8x16;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i8x16 {
        unsafe { i8x16(vreinterpretq_s8_u8(vceqq_s8(self.0, other.0))) }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i8x16 {
        unsafe { i8x16(vreinterpretq_s8_u8(vmvnq_u8(vceqq_s8(self.0, other.0)))) }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i8x16 {
        unsafe { i8x16(vreinterpretq_s8_u8(vcltq_s8(self.0, other.0))) }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i8x16 {
        unsafe {
            i8x16(vreinterpretq_s8_u8(vorrq_u8(
                vcltq_s8(self.0, other.0),
                vceqq_s8(self.0, other.0),
            )))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i8x16 {
        unsafe { i8x16(vreinterpretq_s8_u8(vcgtq_s8(self.0, other.0))) }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i8x16 {
        unsafe {
            i8x16(vreinterpretq_s8_u8(vorrq_u8(
                vcgtq_s8(self.0, other.0),
                vceqq_s8(self.0, other.0),
            )))
        }
    }
}

impl std::ops::Add for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(vaddq_s8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(vsubq_s8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(vmulq_s8(self.0, rhs.0)) }
    }
}
impl std::ops::Div for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                assert!(b[i] != 0, "division by zero");
                result[i] = a[i] / b[i];
            }
            return i8x16(vld1q_s8(result.as_ptr()));
        }
    }
}
impl std::ops::Rem for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] % b[i];
            }
            return i8x16(vld1q_s8(result.as_ptr()));
        }
    }
}
impl std::ops::Neg for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        unsafe { i8x16(vnegq_s8(self.0)) }
    }
}
impl std::ops::BitAnd for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(vandq_s8(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(vorrq_s8(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(veorq_s8(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { i8x16(vmvnq_s8(self.0)) }
    }
}
impl std::ops::Shl for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(vshlq_s8(self.0, rhs.0)) }
    }
}
impl std::ops::Shr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe { i8x16(vshlq_s8(self.0, vnegq_s8(rhs.0))) }
    }
}

impl SimdMath<i8> for i8x16 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { i8x16(vmaxq_s8(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { i8x16(vminq_s8(self.0, other.0)) }
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
        unsafe { i8x16(vabsq_s8(self.0)) }
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
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 16];
            for i in 0..16 {
                result[i] = a[i].pow(b[i] as u32);
            }
            return i8x16(vld1q_s8(result.as_ptr()));
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(i8x16::splat(0)) + alpha * self.min(i8x16::splat(0))
    }
}

impl FloatOutBinary2 for i8x16 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i8")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for i8x16");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                if b[i] < 0 {
                    panic!("Power operation is not supported for negative i8");
                }
                result[i] = a[i].pow(b[i] as u32);
            }
            return i8x16(vld1q_s8(result.as_ptr()));
        }
    }
}
impl Eval2 for i8x16 {
    type Output = i8x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i8x16::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let neq = vmvnq_s8(vreinterpretq_s8_u8(vceqq_s8(self.0, vdupq_n_s8(0))));
            i8x16(vandq_s8(neq, vdupq_n_s8(1)))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i8x16::default()
    }
}
