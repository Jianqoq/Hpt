use std::{
    ops::{Deref, DerefMut},
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::{SimdFloat, SimdInt},
        StdFloat,
    },
};

use crate::{
    traits::{SimdCompare, SimdMath, SimdSelect},
    vectors::traits::VecTrait,
};

use super::i64x4::i64x4;

/// a vector of 4 f64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct f64x4(pub(crate) std::simd::f64x4);

impl Deref for f64x4 {
    type Target = std::simd::f64x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for f64x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl f64x4 {
    /// check if the vector is nan
    pub fn is_nan(&self) -> f64x4 {
        f64x4(self.0.is_nan().to_int().cast())
    }
    /// check if the vector is infinite
    pub fn is_infinite(&self) -> f64x4 {
        f64x4(self.0.is_infinite().to_int().cast())
    }
    /// reciprocal of the vector
    pub fn recip(&self) -> f64x4 {
        f64x4(self.0.recip())
    }
}

impl VecTrait<f64> for f64x4 {
    const SIZE: usize = 4;
    type Base = f64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f64]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        self.as_array().iter().sum()
    }
    fn splat(val: f64) -> f64x4 {
        f64x4(std::simd::f64x4::splat(val))
    }
}

impl SimdSelect<f64x4> for crate::vectors::std_simd::_256bit::i64x4::i64x4 {
    fn select(&self, true_val: f64x4, false_val: f64x4) -> f64x4 {
        let mask: std::simd::mask64x4 = unsafe { std::mem::transmute(*self) };
        f64x4(mask.select(true_val.0, false_val.0))
    }
}

impl SimdCompare for f64x4 {
    type SimdMask = i64x4;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        i64x4(self.0.simd_eq(rhs.0).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        i64x4(self.0.simd_ne(rhs.0).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        i64x4(self.0.simd_lt(rhs.0).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        i64x4(self.0.simd_le(rhs.0).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        i64x4(self.0.simd_gt(rhs.0).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        i64x4(self.0.simd_ge(rhs.0).to_int())
    }
}

impl std::ops::Add for f64x4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        f64x4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for f64x4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        f64x4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for f64x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        f64x4(self.0 * rhs.0)
    }
}
impl std::ops::Div for f64x4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        f64x4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for f64x4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        f64x4(self.0 % rhs.0)
    }
}
impl std::ops::Neg for f64x4 {
    type Output = Self;
    fn neg(self) -> Self {
        f64x4(-self.0)
    }
}

impl SimdMath<f64> for f64x4 {
    fn sin(self) -> Self {
        f64x4(self.0.sin())
    }
    fn cos(self) -> Self {
        f64x4(self.0.cos())
    }
    fn tan(self) -> Self {
        f64x4(sleef::Sleef::tan(self.0))
    }

    fn square(self) -> Self {
        f64x4(self.0 * self.0)
    }

    fn sqrt(self) -> Self {
        f64x4(self.0.sqrt())
    }

    fn abs(self) -> Self {
        f64x4(sleef::Sleef::abs(self.0))
    }

    fn floor(self) -> Self {
        f64x4(self.0.floor())
    }

    fn ceil(self) -> Self {
        f64x4(self.0.ceil())
    }

    fn neg(self) -> Self {
        f64x4(-self.0)
    }

    fn round(self) -> Self {
        f64x4(self.0.round())
    }

    fn sign(self) -> Self {
        unimplemented!()
    }

    fn leaky_relu(self, _: f64) -> Self {
        unimplemented!()
    }

    fn relu(self) -> Self {
        f64x4(sleef::Sleef::max(self.0, f64x4::splat(0.0).0))
    }

    fn relu6(self) -> Self {
        f64x4(sleef::Sleef::min(self.relu().0, f64x4::splat(6.0).0))
    }

    fn pow(self, exp: Self) -> Self {
        f64x4(sleef::Sleef::pow(self.0, exp.0))
    }

    fn asin(self) -> Self {
        f64x4(sleef::Sleef::asin(self.0))
    }

    fn acos(self) -> Self {
        f64x4(sleef::Sleef::acos(self.0))
    }

    fn atan(self) -> Self {
        f64x4(sleef::Sleef::atan(self.0))
    }

    fn sinh(self) -> Self {
        f64x4(sleef::Sleef::sinh(self.0))
    }

    fn cosh(self) -> Self {
        f64x4(sleef::Sleef::cosh(self.0))
    }

    fn tanh(self) -> Self {
        f64x4(sleef::Sleef::tanh(self.0))
    }

    fn asinh(self) -> Self {
        f64x4(sleef::Sleef::asinh(self.0))
    }

    fn acosh(self) -> Self {
        f64x4(sleef::Sleef::acosh(self.0))
    }

    fn atanh(self) -> Self {
        f64x4(sleef::Sleef::atanh(self.0))
    }

    fn exp2(self) -> Self {
        f64x4(sleef::Sleef::exp2(self.0))
    }

    fn exp10(self) -> Self {
        f64x4(sleef::Sleef::exp10(self.0))
    }

    fn expm1(self) -> Self {
        f64x4(sleef::Sleef::exp_m1(self.0))
    }

    fn log10(self) -> Self {
        f64x4(sleef::Sleef::log10(self.0))
    }

    fn log2(self) -> Self {
        f64x4(sleef::Sleef::log2(self.0))
    }

    fn log1p(self) -> Self {
        f64x4(sleef::Sleef::log_1p(self.0))
    }

    fn hypot(self, other: Self) -> Self {
        f64x4(sleef::Sleef::hypot(self.0, other.0))
    }

    fn trunc(self) -> Self {
        f64x4(sleef::Sleef::truncate(self.0))
    }

    fn erf(self) -> Self {
        f64x4(sleef::Sleef::erf(self.0))
    }

    fn cbrt(self) -> Self {
        f64x4(sleef::Sleef::cbrt(self.0))
    }

    fn exp(self) -> Self {
        f64x4(sleef::Sleef::exp(self.0))
    }

    fn ln(self) -> Self {
        f64x4(sleef::Sleef::ln(self.0))
    }

    fn log(self) -> Self {
        f64x4(sleef::Sleef::ln(self.0))
    }

    fn sincos(self) -> (Self, Self) {
        let ret = sleef::Sleef::sin_cos(self.0);
        (f64x4(ret.0), f64x4(ret.1))
    }

    fn atan2(self, other: Self) -> Self {
        f64x4(sleef::Sleef::atan2(self.0, other.0))
    }
}
