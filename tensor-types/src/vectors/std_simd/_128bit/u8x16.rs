use std::{ops::{ Deref, DerefMut }, simd::{cmp::{SimdPartialEq, SimdPartialOrd}, Simd}};

use crate::{impl_std_simd_bit_logic, traits::{SimdCompare, SimdMath, VecTrait}};

use super::i8x16::i8x16;

/// a vector of 16 u8 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct u8x16(pub(crate) std::simd::u8x16);

impl Deref for u8x16 {
    type Target = std::simd::u8x16;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u8x16 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u8> for u8x16 {
    const SIZE: usize = 16;
    type Base = u8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u8]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn sum(&self) -> u8 {
        self.as_array().iter().sum()
    }
    fn splat(val: u8) -> u8x16 {
        u8x16(std::simd::u8x16::splat(val))
    }
}

impl SimdCompare for u8x16 {
    type SimdMask = i8x16;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        i8x16(lhs.simd_eq(rhs).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        i8x16(lhs.simd_ne(rhs).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        i8x16(lhs.simd_lt(rhs).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        i8x16(lhs.simd_le(rhs).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        i8x16(lhs.simd_gt(rhs).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        i8x16(lhs.simd_ge(rhs).to_int())
    }
}

impl std::ops::Add for u8x16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        u8x16(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u8x16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        u8x16(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u8x16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        u8x16(self.0 * rhs.0)
    }
}
impl std::ops::Div for u8x16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        u8x16(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u8x16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        u8x16(self.0 % rhs.0)
    }
}
impl_std_simd_bit_logic!(u8x16);

impl SimdMath<u8> for u8x16 {
    fn max(self, other: Self) -> Self {
        u8x16(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        u8x16(self.0.min(other.0))
    }
    fn relu(self) -> Self {
        u8x16(self.0.max(u8x16::splat(0).0))
    }
    fn relu6(self) -> Self {
        u8x16(self.relu().0.min(u8x16::splat(6).0))
    }
}
