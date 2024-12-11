use crate::traits::{ Init, SimdSelect, VecTrait };
use crate::vectors::arch_simd::_128bit::u32x4::u32x4;
use std::ops::{ Deref, DerefMut };
use std::simd::num::SimdFloat;
use std::simd::StdFloat;

/// a vector of 4 f32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
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
}
impl Init<f32> for f32x4 {
    fn splat(val: f32) -> f32x4 {
        f32x4(std::simd::f32x4::splat(val))
    }
}
impl SimdSelect<f32x4> for u32x4 {
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