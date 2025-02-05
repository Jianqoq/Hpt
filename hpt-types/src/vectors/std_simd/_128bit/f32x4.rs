use crate::traits::{SimdCompare, SimdMath, SimdSelect, VecTrait};
use std::ops::{Deref, DerefMut};
use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::StdFloat;

use super::i32x4::i32x4;

/// a vector of 4 f32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct f32x4(pub(crate) std::simd::f32x4);

impl Deref for f32x4 {
    type Target = std::simd::f32x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for f32x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl f32x4 {
    /// check if the vector is nan
    pub fn is_nan(&self) -> f32x4 {
        f32x4(self.0.is_nan().to_int().cast())
    }
    /// check if the vector is infinite
    pub fn is_infinite(&self) -> f32x4 {
        f32x4(self.0.is_infinite().to_int().cast())
    }
    /// reciprocal of the vector
    pub fn recip(&self) -> f32x4 {
        f32x4(self.0.recip())
    }
}

impl VecTrait<f32> for f32x4 {
    const SIZE: usize = 4;
    type Base = f32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        self.as_mut_array().copy_from_slice(slice)
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32x4(self.0.mul_add(a.0, b.0))
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        self.reduce_sum()
    }
    fn splat(val: f32) -> f32x4 {
        f32x4(std::simd::f32x4::splat(val))
    }
}

impl SimdCompare for f32x4 {
    type SimdMask = i32x4;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_eq(rhs.0).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_ne(rhs.0).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_lt(rhs.0).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_le(rhs.0).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_gt(rhs.0).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_ge(rhs.0).to_int())
    }
}

impl SimdSelect<f32x4> for i32x4 {
    fn select(&self, true_val: f32x4, false_val: f32x4) -> f32x4 {
        let mask: std::simd::mask32x4 = unsafe { std::mem::transmute(*self) };
        f32x4(mask.select(true_val.0, false_val.0))
    }
}
impl std::ops::Add for f32x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        f32x4(self.0 + rhs.0)
    }
}

impl std::ops::Sub for f32x4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        f32x4(self.0 - rhs.0)
    }
}

impl std::ops::Mul for f32x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        f32x4(self.0 * rhs.0)
    }
}

impl std::ops::Div for f32x4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        f32x4(self.0 / rhs.0)
    }
}

impl std::ops::Rem for f32x4 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        f32x4(self.0 % rhs.0)
    }
}
impl std::ops::Neg for f32x4 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        f32x4(-self.0)
    }
}

impl SimdMath<f32> for f32x4 {
    fn sin(self) -> Self {
        f32x4(self.0.sin())
    }
    fn cos(self) -> Self {
        f32x4(self.0.cos())
    }
    fn tan(self) -> Self {
        f32x4(sleef::Sleef::tan(self.0))
    }

    fn square(self) -> Self {
        f32x4(self.0 * self.0)
    }

    fn sqrt(self) -> Self {
        f32x4(sleef::Sleef::sqrt(self.0))
    }

    fn abs(self) -> Self {
        f32x4(sleef::Sleef::abs(self.0))
    }

    fn floor(self) -> Self {
        f32x4(self.0.floor())
    }

    fn ceil(self) -> Self {
        f32x4(self.0.ceil())
    }

    fn neg(self) -> Self {
        f32x4(-self.0)
    }

    fn round(self) -> Self {
        f32x4(self.0.round())
    }

    fn sign(self) -> Self {
        todo!()
    }

    fn leaky_relu(self, _: f32) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        f32x4(sleef::Sleef::max(self.0, Self::splat(0.0).0))
    }

    fn relu6(self) -> Self {
        f32x4(sleef::Sleef::min(self.relu().0, Self::splat(6.0).0))
    }

    fn pow(self, exp: Self) -> Self {
        f32x4(sleef::Sleef::pow(self.0, exp.0))
    }

    fn asin(self) -> Self {
        f32x4(sleef::Sleef::asin(self.0))
    }

    fn acos(self) -> Self {
        f32x4(sleef::Sleef::acos(self.0))
    }

    fn atan(self) -> Self {
        f32x4(sleef::Sleef::atan(self.0))
    }

    fn sinh(self) -> Self {
        f32x4(sleef::Sleef::sinh(self.0))
    }

    fn cosh(self) -> Self {
        f32x4(sleef::Sleef::cosh(self.0))
    }

    fn tanh(self) -> Self {
        f32x4(sleef::Sleef::tanh(self.0))
    }

    fn asinh(self) -> Self {
        f32x4(sleef::Sleef::asinh(self.0))
    }

    fn acosh(self) -> Self {
        f32x4(sleef::Sleef::acosh(self.0))
    }

    fn atanh(self) -> Self {
        f32x4(sleef::Sleef::atanh(self.0))
    }

    fn exp2(self) -> Self {
        f32x4(sleef::Sleef::exp2(self.0))
    }

    fn exp10(self) -> Self {
        f32x4(sleef::Sleef::exp10(self.0))
    }

    fn expm1(self) -> Self {
        f32x4(sleef::Sleef::exp_m1(self.0))
    }

    fn log10(self) -> Self {
        f32x4(sleef::Sleef::log10(self.0))
    }

    fn log2(self) -> Self {
        f32x4(sleef::Sleef::log2(self.0))
    }

    fn log1p(self) -> Self {
        f32x4(sleef::Sleef::log_1p(self.0))
    }

    fn hypot(self, other: Self) -> Self {
        f32x4(sleef::Sleef::hypot(self.0, other.0))
    }

    fn trunc(self) -> Self {
        f32x4(self.0.trunc())
    }

    fn erf(self) -> Self {
        f32x4(sleef::Sleef::erf(self.0))
    }

    fn cbrt(self) -> Self {
        f32x4(sleef::Sleef::cbrt(self.0))
    }

    fn exp(self) -> Self {
        f32x4(sleef::Sleef::exp(self.0))
    }

    fn ln(self) -> Self {
        f32x4(sleef::Sleef::ln(self.0))
    }

    fn log(self) -> Self {
        f32x4(sleef::Sleef::ln(self.0))
    }

    fn sincos(self) -> (Self, Self) {
        let (x, y) = sleef::Sleef::sin_cos(self.0);
        (f32x4(x), f32x4(y))
    }

    fn atan2(self, other: Self) -> Self {
        f32x4(sleef::Sleef::atan2(self.0, other.0))
    }

    fn min(self, other: Self) -> Self {
        f32x4(sleef::Sleef::min(self.0, other.0))
    }

    fn max(self, other: Self) -> Self {
        f32x4(sleef::Sleef::max(self.0, other.0))
    }
}
