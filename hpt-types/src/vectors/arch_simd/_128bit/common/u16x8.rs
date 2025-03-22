use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait},
    type_promote::{NormalOut2, NormalOutUnary2},
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i16x8::i16x8;
use crate::arch_simd::_128bit::common::bf16x8::bf16x8;

/// a vector of 8 u16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u16x8(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")] pub(crate) uint16x8_t,
);

#[allow(non_camel_case_types)]
pub(crate) type u16_promote = u16x8;

impl SimdCompare for u16x8 {
    type SimdMask = i16x8;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }

    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }

    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }

    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }

    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl VecConvertor for u16x8 {
    #[inline(always)]
    fn to_u16(self) -> u16x8 {
        self
    }
    #[inline(always)]
    fn to_i16(self) -> i16x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f16(self) -> super::f16x8::f16x8 {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let mut result = [half::f16::ZERO; 8];
            for i in 0..8 {
                result[i] = half::f16::from_f32(arr[i] as f32);
            }
            super::f16x8::f16x8(result)
        }
    }
    #[inline(always)]
    fn to_bf16(self) -> bf16x8 {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let mut result = [half::bf16::ZERO; 8];
            for i in 0..8 {
                result[i] = half::bf16::from_f32(arr[i] as f32);
            }
            bf16x8(result)
        }
    }
}

impl NormalOut2 for u16x8 {
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

impl NormalOutUnary2 for u16x8 {
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
        self.max(u16x8::splat(0)) + alpha * self.min(u16x8::splat(0))
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
