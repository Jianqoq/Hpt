use crate::{
    convertion::VecConvertor,
    traits::{SimdMath, VecTrait},
    type_promote::NormalOut2,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::u16x32::u16x32;

/// a vector of 16 i16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct i16x32(#[cfg(target_arch = "x86_64")] pub(crate) __m512i);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type i16_promote = i16x32;

impl VecConvertor for i16x32 {
    #[inline(always)]
    fn to_i16(self) -> i16x32 {
        self
    }
    #[inline(always)]
    fn to_u16(self) -> u16x32 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f16(self) -> super::f16x32::f16x32 {
        let mut result = [half::f16::ZERO; 16];
        let arr: [i16; 32] = unsafe { std::mem::transmute(self.0) };
        for i in 0..32 {
            result[i] = half::f16::from_f32(arr[i] as f32);
        }
        super::f16x32::f16x32(result)
    }
}

impl NormalOut2 for i16x32 {
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
