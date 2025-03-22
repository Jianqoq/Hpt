use crate::{
    convertion::VecConvertor,
    traits::{SimdMath, VecTrait},
    type_promote::NormalOut2,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::u16x16::u16x16;

/// a vector of 16 i16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct i16x16(#[cfg(target_arch = "x86_64")] pub(crate) __m256i);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type i16_promote = i16x16;

impl VecConvertor for i16x16 {
    #[inline(always)]
    fn to_i16(self) -> i16x16 {
        self
    }
    #[inline(always)]
    fn to_u16(self) -> u16x16 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f16(self) -> super::f16x16::f16x16 {
        let mut result = [half::f16::ZERO; 16];
        let arr: [i16; 16] = unsafe { std::mem::transmute(self.0) };
        for i in 0..16 {
            result[i] = half::f16::from_f32(arr[i] as f32);
        }
        super::f16x16::f16x16(result)
    }
}

impl NormalOut2 for i16x16 {
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
