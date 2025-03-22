use crate::{
    convertion::VecConvertor,
    traits::{SimdMath, VecTrait},
    type_promote::NormalOut2,
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::vectors::arch_simd::_128bit::u16x8;

/// a vector of 8 i16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct i16x8(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")] pub(crate) int16x8_t,
);

#[allow(non_camel_case_types)]
pub(crate) type i16_promote = i16x8;

impl VecConvertor for i16x8 {
    #[inline(always)]
    fn to_i16(self) -> i16x8 {
        self
    }
    #[inline(always)]
    fn to_u16(self) -> u16x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f16(self) -> super::f16x8::f16x8 {
        let mut result = [half::f16::ZERO; 8];
        let arr: [i16; 8] = unsafe { std::mem::transmute(self.0) };
        for i in 0..8 {
            result[i] = half::f16::from_f32(arr[i] as f32);
        }
        super::f16x8::f16x8(result)
    }
}

impl NormalOut2 for i16x8 {
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
