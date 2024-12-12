use crate::vectors::traits::{SimdSelect, VecTrait};
use std::ops::{Deref, DerefMut};
use std::simd::num::SimdFloat;
use std::simd::StdFloat;

/// a vector of 8 f32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct f32x8(pub(crate) std::simd::f32x8);

impl Deref for f32x8 {
    type Target = std::simd::f32x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for f32x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<f32> for f32x8 {
    const SIZE: usize = 8;
    type Base = f32;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32x8(self.0.mul_add(a.0, b.0))
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        self.as_mut_array().copy_from_slice(slice)
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        self.reduce_sum()
    }
    fn splat(val: f32) -> f32x8 {
        f32x8(std::simd::f32x8::splat(val))
    }
}

impl SimdSelect<f32x8> for crate::vectors::std_simd::_256bit::u32x8::u32x8 {
    fn select(&self, true_val: f32x8, false_val: f32x8) -> f32x8 {
        let mask: std::simd::mask32x8 = unsafe { std::mem::transmute(*self) };
        f32x8(mask.select(true_val.0, false_val.0))
    }
}

impl std::ops::Add for f32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        f32x8(self.0 + rhs.0)
    }
}

impl std::ops::Sub for f32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        f32x8(self.0 - rhs.0)
    }
}

impl std::ops::Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        f32x8(self.0 * rhs.0)
    }
}

impl std::ops::Div for f32x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        f32x8(self.0 / rhs.0)
    }
}

impl std::ops::Rem for f32x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        f32x8(self.0 % rhs.0)
    }
}
impl std::ops::Neg for f32x8 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        f32x8(-self.0)
    }
}

impl SimdMath<f32> for f32x8 {
    fn sin(self) -> Self {
        f32x8(unsafe { xsinf_u1(self.0) })
    }
    fn cos(self) -> Self {
        f32x8(unsafe { xcosf_u1(self.0) })
    }
    fn tan(self) -> Self {
        f32x8(unsafe { xtanf_u1(self.0) })
    }

    fn square(self) -> Self {
        f32x8(unsafe { _mm256_mul_ps(self.0, self.0) })
    }

    fn sqrt(self) -> Self {
        f32x8(unsafe { xsqrtf_u05(self.0) })
    }

    fn abs(self) -> Self {
        f32x8(unsafe { vabs_vf_vf(self.0) })
    }

    fn floor(self) -> Self {
        f32x8(unsafe { _mm256_floor_ps(self.0) })
    }

    fn ceil(self) -> Self {
        f32x8(unsafe { _mm256_ceil_ps(self.0) })
    }

    fn neg(self) -> Self {
        f32x8(unsafe { _mm256_sub_ps(_mm256_setzero_ps(), self.0) })
    }

    fn round(self) -> Self {
        f32x8(unsafe { xroundf(self.0) })
    }

    fn sign(self) -> Self {
        f32x8(unsafe { _mm256_and_ps(self.0, _mm256_set1_ps(0.0f32)) })
    }

    fn leaky_relu(self, _: f32) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        f32x8(unsafe { _mm256_max_ps(self.0, _mm256_setzero_ps()) })
    }

    fn relu6(self) -> Self {
        f32x8(unsafe { _mm256_min_ps(self.relu().0, _mm256_set1_ps(6.0f32)) })
    }

    fn pow(self, exp: Self) -> Self {
        f32x8(unsafe { xpowf(self.0, exp.0) })
    }

    fn asin(self) -> Self {
        f32x8(unsafe { xasinf_u1(self.0) })
    }

    fn acos(self) -> Self {
        f32x8(unsafe { xacosf_u1(self.0) })
    }

    fn atan(self) -> Self {
        f32x8(unsafe { xatanf_u1(self.0) })
    }

    fn sinh(self) -> Self {
        f32x8(unsafe { xsinhf(self.0) })
    }

    fn cosh(self) -> Self {
        f32x8(unsafe { xcoshf(self.0) })
    }

    fn tanh(self) -> Self {
        f32x8(unsafe { xtanhf(self.0) })
    }

    fn asinh(self) -> Self {
        f32x8(unsafe { xasinhf(self.0) })
    }

    fn acosh(self) -> Self {
        f32x8(unsafe { xacoshf(self.0) })
    }

    fn atanh(self) -> Self {
        f32x8(unsafe { xatanhf(self.0) })
    }

    fn exp2(self) -> Self {
        f32x8(unsafe { xexp2f(self.0) })
    }

    fn exp10(self) -> Self {
        f32x8(unsafe { xexp10f(self.0) })
    }

    fn expm1(self) -> Self {
        f32x8(unsafe { xexpm1f(self.0) })
    }

    fn log10(self) -> Self {
        f32x8(unsafe { xlog10f(self.0) })
    }

    fn log2(self) -> Self {
        f32x8(unsafe { xlog2f(self.0) })
    }

    fn log1p(self) -> Self {
        f32x8(unsafe { xlog1pf(self.0) })
    }

    fn hypot(self, other: Self) -> Self {
        f32x8(unsafe { xhypotf_u05(self.0, other.0) })
    }

    fn trunc(self) -> Self {
        f32x8(unsafe { xtruncf(self.0) })
    }

    fn erf(self) -> Self {
        f32x8(unsafe { xerff_u1(self.0) })
    }

    fn cbrt(self) -> Self {
        f32x8(unsafe { xcbrtf_u1(self.0) })
    }

    fn exp(self) -> Self {
        f32x8(unsafe { xexpf(self.0) })
    }

    fn ln(self) -> Self {
        f32x8(unsafe { xlogf_u1(self.0) })
    }

    fn log(self) -> Self {
        f32x8(unsafe { xlogf_u1(self.0) })
    }

    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincosf_u1(self.0) };
        (f32x8(ret.x), f32x8(ret.y))
    }

    fn atan2(self, other: Self) -> Self {
        f32x8(unsafe { xatan2f_u1(self.0, other.0) })
    }
}
