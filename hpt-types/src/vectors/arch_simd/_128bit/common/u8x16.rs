use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait},
    type_promote::{NormalOut2, NormalOutUnary2},
};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i8x16::i8x16;
use crate::vectors::arch_simd::_128bit::common::boolx16::boolx16;

/// a vector of 16 u8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u8x16(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")] pub(crate) uint8x16_t,
);

#[allow(non_camel_case_types)]
pub(crate) type u8_promote = u8x16;

impl SimdCompare for u8x16 {
    type SimdMask = i8x16;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl VecConvertor for u8x16 {
    #[inline(always)]
    fn to_u8(self) -> u8x16 {
        self
    }
    #[inline(always)]
    fn to_i8(self) -> i8x16 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_bool(self) -> boolx16 {
        unsafe { std::mem::transmute(self) }
    }
}

impl NormalOut2 for u8x16 {
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

impl NormalOutUnary2 for u8x16 {
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
    fn __trunc(self) -> Self {
        self
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(u8x16::splat(0)) + alpha * self.min(u8x16::splat(0))
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
    fn __copysign(self, rhs: Self) -> Self {
        self.abs() * rhs.signum()
    }
}
