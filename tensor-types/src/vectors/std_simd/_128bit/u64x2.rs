use std::{ops::{ Deref, DerefMut }, simd::{cmp::{SimdPartialEq, SimdPartialOrd}, Simd}};

use crate::{impl_std_simd_bit_logic, traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait}};

use super::i64x2::i64x2;

/// a vector of 2 u64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct u64x2(pub(crate) std::simd::u64x2);

impl Deref for u64x2 {
    type Target = std::simd::u64x2;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u64x2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u64> for u64x2 {
    const SIZE: usize = 2;
    type Base = u64;
    
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u64]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn sum(&self) -> u64 {
        self.as_array().iter().sum()
    }
    fn splat(val: u64) -> u64x2 {
        u64x2(std::simd::u64x2::splat(val))
    }
}

impl SimdCompare for u64x2 {
    type SimdMask = i64x2;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 2> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 2> = unsafe { std::mem::transmute(rhs) };
        i64x2(lhs.simd_eq(rhs).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 2> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 2> = unsafe { std::mem::transmute(rhs) };
        i64x2(lhs.simd_ne(rhs).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 2> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 2> = unsafe { std::mem::transmute(rhs) };
        i64x2(lhs.simd_lt(rhs).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 2> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 2> = unsafe { std::mem::transmute(rhs) };
        i64x2(lhs.simd_le(rhs).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 2> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 2> = unsafe { std::mem::transmute(rhs) };
        i64x2(lhs.simd_gt(rhs).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u64, 2> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u64, 2> = unsafe { std::mem::transmute(rhs) };
        i64x2(lhs.simd_ge(rhs).to_int())
    }
}

impl SimdSelect<u64x2> for u64x2 {
    fn select(&self, true_val: u64x2, false_val: u64x2) -> u64x2 {
        let mask: std::simd::mask64x2 = unsafe { std::mem::transmute(*self) };
        u64x2(mask.select(true_val.0, false_val.0))
    }
}

impl std::ops::Add for u64x2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        u64x2(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u64x2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        u64x2(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u64x2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        u64x2(self.0 * rhs.0)
    }
}
impl std::ops::Div for u64x2 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        u64x2(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u64x2 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        u64x2(self.0 % rhs.0)
    }
}
impl_std_simd_bit_logic!(u64x2);

impl SimdMath<u64> for u64x2 {
    fn max(self, other: Self) -> Self {
        u64x2(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        u64x2(self.0.min(other.0))
    }
    fn relu(self) -> Self {
        u64x2(self.0.max(u64x2::splat(0).0))
    }
    fn relu6(self) -> Self {
        u64x2(self.relu().0.min(u64x2::splat(6).0))
    }
}