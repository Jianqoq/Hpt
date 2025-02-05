use std::{
    ops::{Deref, DerefMut},
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdUint,
        Simd,
    },
};

use crate::{
    impl_std_simd_bit_logic,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};

use super::i16x8::i16x8;

/// a vector of 8 u16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct u16x8(pub(crate) std::simd::u16x8);

impl Deref for u16x8 {
    type Target = std::simd::u16x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u16x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u16> for u16x8 {
    const SIZE: usize = 8;
    type Base = u16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u16]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        self.as_array().iter().sum()
    }
    fn splat(val: u16) -> u16x8 {
        u16x8(std::simd::u16x8::splat(val))
    }
}

impl SimdCompare for u16x8 {
    type SimdMask = i16x8;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u16, 8> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u16, 8> = unsafe { std::mem::transmute(rhs) };
        i16x8(lhs.simd_eq(rhs).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u16, 8> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u16, 8> = unsafe { std::mem::transmute(rhs) };
        i16x8(lhs.simd_ne(rhs).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u16, 8> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u16, 8> = unsafe { std::mem::transmute(rhs) };
        i16x8(lhs.simd_lt(rhs).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u16, 8> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u16, 8> = unsafe { std::mem::transmute(rhs) };
        i16x8(lhs.simd_le(rhs).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u16, 8> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u16, 8> = unsafe { std::mem::transmute(rhs) };
        i16x8(lhs.simd_gt(rhs).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<u16, 8> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u16, 8> = unsafe { std::mem::transmute(rhs) };
        i16x8(lhs.simd_ge(rhs).to_int())
    }
}

impl SimdSelect<u16x8> for u16x8 {
    fn select(&self, true_val: u16x8, false_val: u16x8) -> u16x8 {
        let mask: std::simd::mask16x8 = unsafe { std::mem::transmute(*self) };
        u16x8(mask.select(true_val.0, false_val.0))
    }
}

impl std::ops::Add for u16x8 {
    type Output = u16x8;
    fn add(self, rhs: Self) -> Self::Output {
        u16x8(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u16x8 {
    type Output = u16x8;
    fn sub(self, rhs: Self) -> Self::Output {
        u16x8(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u16x8 {
    type Output = u16x8;
    fn mul(self, rhs: Self) -> Self::Output {
        u16x8(self.0 * rhs.0)
    }
}
impl std::ops::Div for u16x8 {
    type Output = u16x8;
    fn div(self, rhs: Self) -> Self::Output {
        u16x8(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u16x8 {
    type Output = u16x8;
    fn rem(self, rhs: Self) -> Self::Output {
        u16x8(self.0 % rhs.0)
    }
}
impl_std_simd_bit_logic!(u16x8);

impl SimdMath<u16> for u16x8 {
    fn max(self, other: Self) -> Self {
        u16x8(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        u16x8(self.0.min(other.0))
    }
    fn relu(self) -> Self {
        u16x8(self.0.max(u16x8::splat(0).0))
    }
    fn relu6(self) -> Self {
        u16x8(self.relu().0.min(u16x8::splat(6).0))
    }
    fn neg(self) -> Self {
        u16x8(self.0.wrapping_neg())
    }
}
