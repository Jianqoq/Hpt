use std::{ops::{ Deref, DerefMut }, simd::{cmp::{SimdPartialEq, SimdPartialOrd}, num::SimdUint, Simd}};

use crate::{impl_std_simd_bit_logic, traits::{SimdCompare, SimdMath}, vectors::traits::VecTrait};

use super::i64x4::i64x4;

/// a vector of 4 u64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct u64x4(pub(crate) std::simd::u64x4);

impl Deref for u64x4 {
    type Target = std::simd::u64x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u64x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u64> for u64x4 {
    const SIZE: usize = 4;
    type Base = u64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u64]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn sum(&self) -> u64 {
        self.as_array().iter().sum()
    }
    fn splat(val: u64) -> u64x4 {
        u64x4(std::simd::u64x4::splat(val))
    }
}

impl SimdCompare for u64x4 {
    type SimdMask = i64x4;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 4> = unsafe { std::mem::transmute(rhs) };
        i64x4(lhs.simd_eq(rhs).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 4> = unsafe { std::mem::transmute(rhs) };
        i64x4(lhs.simd_ne(rhs).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 4> = unsafe { std::mem::transmute(rhs) };
        i64x4(lhs.simd_lt(rhs).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 4> = unsafe { std::mem::transmute(rhs) };
        i64x4(lhs.simd_le(rhs).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 4> = unsafe { std::mem::transmute(rhs) };
        i64x4(lhs.simd_gt(rhs).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 4> = unsafe { std::mem::transmute(rhs) };
        i64x4(lhs.simd_ge(rhs).to_int())
    }
}

impl std::ops::Add for u64x4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        u64x4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u64x4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        u64x4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u64x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        u64x4(self.0 * rhs.0)
    }
}
impl std::ops::Div for u64x4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        u64x4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u64x4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        u64x4(self.0 % rhs.0)
    }
}

impl_std_simd_bit_logic!(u64x4);

impl SimdMath<u64> for u64x4 {
    fn max(self, other: Self) -> Self {
        u64x4(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        u64x4(self.0.min(other.0))
    }
    fn relu(self) -> Self {
        u64x4(self.0.max(u64x4::splat(0).0))
    }
    fn relu6(self) -> Self {
        u64x4(self.relu().0.min(u64x4::splat(6).0))
    }
    fn neg(self) -> Self {
        u64x4(self.0.wrapping_neg())
    }
}