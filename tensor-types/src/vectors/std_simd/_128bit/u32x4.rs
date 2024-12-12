use std::{ops::{ Deref, DerefMut }, simd::{cmp::{SimdPartialEq, SimdPartialOrd}, Simd}};

use crate::{impl_std_simd_bit_logic, traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait }};

use super::i32x4::i32x4;

/// a vector of 4 u32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct u32x4(pub(crate) std::simd::u32x4);

impl Deref for u32x4 {
    type Target = std::simd::u32x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u32x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u32> for u32x4 {
    const SIZE: usize = 4;
    type Base = u32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u32]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn sum(&self) -> u32 {
        self.as_array().iter().sum()
    }
    fn splat(val: u32) -> u32x4 {
        u32x4(std::simd::u32x4::splat(val))
    }
}

impl SimdCompare for u32x4 {
    type SimdMask = i32x4;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u32, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u32, 4> = unsafe { std::mem::transmute(rhs) };
        i32x4(lhs.simd_eq(rhs).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u32, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u32, 4> = unsafe { std::mem::transmute(rhs) };
        i32x4(lhs.simd_ne(rhs).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u32, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u32, 4> = unsafe { std::mem::transmute(rhs) };
        i32x4(lhs.simd_lt(rhs).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u32, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u32, 4> = unsafe { std::mem::transmute(rhs) };
        i32x4(lhs.simd_le(rhs).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u32, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u32, 4> = unsafe { std::mem::transmute(rhs) };
        i32x4(lhs.simd_gt(rhs).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u32, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u32, 4> = unsafe { std::mem::transmute(rhs) };
        i32x4(lhs.simd_ge(rhs).to_int())
    }
}

impl SimdSelect<u32x4> for u32x4 {
    fn select(&self, true_val: u32x4, false_val: u32x4) -> u32x4 {
        let mask: std::simd::mask32x4 = unsafe { std::mem::transmute(*self) };
        u32x4(mask.select(true_val.0, false_val.0))
    }
}

impl std::ops::Add for u32x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        u32x4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u32x4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        u32x4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u32x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        u32x4(self.0 * rhs.0)
    }
}
impl std::ops::Div for u32x4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        u32x4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u32x4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        u32x4(self.0 % rhs.0)
    }
}
impl_std_simd_bit_logic!(u32x4);

impl SimdMath<u32> for u32x4 {
    fn max(self, other: Self) -> Self {
        u32x4(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        u32x4(self.0.min(other.0))
    }
    fn relu(self) -> Self {
        u32x4(self.0.max(u32x4::splat(0).0))
    }
    fn relu6(self) -> Self {
        u32x4(self.relu().0.min(u32x4::splat(6).0))
    }
}
