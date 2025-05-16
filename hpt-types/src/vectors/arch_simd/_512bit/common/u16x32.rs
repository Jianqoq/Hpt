use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait},
    type_promote::{NormalOut2, NormalOutUnary2},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i16x32::i16x32;

/// a vector of 16 u16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u16x32(#[cfg(target_arch = "x86_64")] pub(crate) __m512i);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type u16_promote = u16x32;

impl SimdCompare for u16x32 {
    type SimdMask = i16x32;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x32 = std::mem::transmute(self.0);
            let rhs: i16x32 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x32 = std::mem::transmute(self.0);
            let rhs: i16x32 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x32 = std::mem::transmute(self.0);
            let rhs: i16x32 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x32 = std::mem::transmute(self.0);
            let rhs: i16x32 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x32 = std::mem::transmute(self.0);
            let rhs: i16x32 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x32 = std::mem::transmute(self.0);
            let rhs: i16x32 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl VecConvertor for u16x32 {
    #[inline(always)]
    fn to_u16(self) -> u16x32 {
        self
    }
    #[inline(always)]
    fn to_i16(self) -> i16x32 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f16(self) -> super::f16x32::f16x32 {
        unsafe {
            let arr: [u16; 32] = std::mem::transmute(self.0);
            let mut result = [half::f16::ZERO; 32];
            for i in 0..32 {
                result[i] = half::f16::from_f32(arr[i] as f32);
            }
            super::f16x32::f16x32(result)
        }
    }
    #[inline(always)]
    fn to_bf16(self) -> super::bf16x32::bf16x32 {
        unsafe {
            let arr: [u16; 32] = std::mem::transmute(self.0);
            let mut result = [half::bf16::ZERO; 32];
            for i in 0..32 {
                result[i] = half::bf16::from_f32(arr[i] as f32);
            }
            super::bf16x32::bf16x32(result)
        }
    }
}

impl NormalOut2 for u16x32 {
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

impl NormalOutUnary2 for u16x32 {
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
