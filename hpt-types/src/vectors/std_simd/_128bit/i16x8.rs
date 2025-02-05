use std::{
    ops::{Deref, DerefMut},
    simd::cmp::{SimdPartialEq, SimdPartialOrd},
};

use crate::{
    impl_std_simd_bit_logic,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};

/// a vector of 8 i16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct i16x8(pub(crate) std::simd::i16x8);

impl Deref for i16x8 {
    type Target = std::simd::i16x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i16x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i16> for i16x8 {
    const SIZE: usize = 8;
    type Base = i16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i16]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn sum(&self) -> i16 {
        self.as_array().iter().sum()
    }
    fn splat(val: i16) -> i16x8 {
        i16x8(std::simd::i16x8::splat(val))
    }
}

impl SimdCompare for i16x8 {
    type SimdMask = i16x8;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        i16x8(self.0.simd_eq(rhs.0).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        i16x8(self.0.simd_ne(rhs.0).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        i16x8(self.0.simd_lt(rhs.0).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        i16x8(self.0.simd_le(rhs.0).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        i16x8(self.0.simd_gt(rhs.0).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        i16x8(self.0.simd_ge(rhs.0).to_int())
    }
}

impl SimdSelect<i16x8> for crate::vectors::std_simd::_128bit::i16x8::i16x8 {
    fn select(&self, true_val: i16x8, false_val: i16x8) -> i16x8 {
        let mask: std::simd::mask16x8 = unsafe { std::mem::transmute(*self) };
        i16x8(mask.select(true_val.0, false_val.0))
    }
}

impl std::ops::Add for i16x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        i16x8(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i16x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        i16x8(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i16x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        i16x8(self.0 * rhs.0)
    }
}
impl std::ops::Div for i16x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        i16x8(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i16x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        i16x8(self.0 % rhs.0)
    }
}
impl std::ops::Neg for i16x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        i16x8(-self.0)
    }
}
impl_std_simd_bit_logic!(i16x8);

impl SimdMath<i16> for i16x8 {
    fn max(self, other: Self) -> Self {
        i16x8(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        i16x8(self.0.min(other.0))
    }
    fn relu(self) -> Self {
        i16x8(self.0.max(i16x8::splat(0).0))
    }
    fn relu6(self) -> Self {
        i16x8(self.relu().0.min(i16x8::splat(6).0))
    }
}
