use std::{ops::{ Deref, DerefMut }, simd::cmp::{SimdPartialEq, SimdPartialOrd}};

use crate::{impl_std_simd_bit_logic, traits::{SimdCompare, SimdMath}, vectors::traits::VecTrait};

/// a vector of 32 i8 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct i8x32(pub(crate) std::simd::i8x32);

impl Deref for i8x32 {
    type Target = std::simd::i8x32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i8x32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i8> for i8x32 {
    const SIZE: usize = 32;
    type Base = i8;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i8]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        self.as_array().iter().sum()
    }
    fn splat(val: i8) -> i8x32 {
        i8x32(std::simd::i8x32::splat(val))
    }
}

impl SimdCompare for i8x32 {
    type SimdMask = i8x32;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        i8x32(self.0.simd_eq(rhs.0).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        i8x32(self.0.simd_ne(rhs.0).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        i8x32(self.0.simd_lt(rhs.0).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        i8x32(self.0.simd_le(rhs.0).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        i8x32(self.0.simd_gt(rhs.0).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        i8x32(self.0.simd_ge(rhs.0).to_int())
    }
}

impl std::ops::Add for i8x32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        i8x32(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i8x32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        i8x32(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i8x32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        i8x32(self.0 * rhs.0)
    }
}
impl std::ops::Div for i8x32 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        i8x32(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i8x32 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        i8x32(self.0 % rhs.0)
    }
}
impl std::ops::Neg for i8x32 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        i8x32(-self.0)
    }
}

impl_std_simd_bit_logic!(i8x32);

impl SimdMath<i8> for i8x32 {
    fn max(self, other: Self) -> Self {
        i8x32(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        i8x32(self.0.min(other.0))
    }
    fn relu(self) -> Self {
        i8x32(self.0.max(i8x32::splat(0).0))
    }
    fn relu6(self) -> Self {
        i8x32(self.relu().0.min(i8x32::splat(6).0))
    }
}
