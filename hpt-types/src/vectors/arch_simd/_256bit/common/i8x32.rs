use crate::{
    convertion::VecConvertor,
    traits::{SimdMath, VecTrait},
    type_promote::NormalOut2,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::u8x32::u8x32;

/// a vector of 32 i8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct i8x32(#[cfg(target_arch = "x86_64")] pub(crate) __m256i);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type i8_promote = i8x32;

impl VecConvertor for i8x32 {
    #[inline(always)]
    fn to_i8(self) -> i8x32 {
        self
    }
    #[inline(always)]
    fn to_u8(self) -> u8x32 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_bool(self) -> super::boolx32::boolx32 {
        unsafe { std::mem::transmute(self) }
    }
}

impl NormalOut2 for i8x32 {
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
