use std::{ ops::{ Deref, DerefMut }, simd::StdFloat };

use crate::vectors::traits::VecTrait;

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
