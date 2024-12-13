use std::{ops::{ Deref, DerefMut }, simd::cmp::{SimdPartialEq, SimdPartialOrd}};

use crate::{impl_std_simd_bit_logic, traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait }};

/// a vector of 2 i64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct i64x2(pub(crate) std::simd::i64x2);

impl Deref for i64x2 {
    type Target = std::simd::i64x2;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i64x2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i64> for i64x2 {
    const SIZE: usize = 2;
    type Base = i64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i64]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn sum(&self) -> i64 {
        self.as_array().iter().sum()
    }
    fn splat(val: i64) -> i64x2 {
        i64x2(std::simd::i64x2::splat(val))
    }
}

impl SimdCompare for i64x2 {
    type SimdMask = i64x2;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        i64x2(self.0.simd_eq(rhs.0).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        i64x2(self.0.simd_ne(rhs.0).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        i64x2(self.0.simd_lt(rhs.0).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        i64x2(self.0.simd_le(rhs.0).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        i64x2(self.0.simd_gt(rhs.0).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        i64x2(self.0.simd_ge(rhs.0).to_int())
    }
}

impl SimdSelect<i64x2> for crate::vectors::std_simd::_128bit::i64x2::i64x2 {
    fn select(&self, true_val: i64x2, false_val: i64x2) -> i64x2 {
        let mask: std::simd::mask64x2 = unsafe { std::mem::transmute(*self) };
        i64x2(mask.select(true_val.0, false_val.0))
    }
}

impl std::ops::Add for i64x2 {
    type Output = i64x2;
    fn add(self, rhs: Self) -> Self::Output {
        i64x2(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i64x2 {
    type Output = i64x2;
    fn sub(self, rhs: Self) -> Self::Output {
        i64x2(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i64x2 {
    type Output = i64x2;
    fn mul(self, rhs: Self) -> Self::Output {
        i64x2(self.0 * rhs.0)
    }
}
impl std::ops::Div for i64x2 {
    type Output = i64x2;
    fn div(self, rhs: Self) -> Self::Output {
        i64x2(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i64x2 {
    type Output = i64x2;
    fn rem(self, rhs: Self) -> Self::Output {
        i64x2(self.0 % rhs.0)
    }
}
impl std::ops::Neg for i64x2 {
    type Output = i64x2;
    fn neg(self) -> Self::Output {
        i64x2(-self.0)
    }
}
impl_std_simd_bit_logic!(i64x2);

impl SimdMath<i64> for i64x2 {
    fn max(self, other: Self) -> Self {
        i64x2(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        i64x2(self.0.min(other.0))
    }
    fn relu(self) -> Self {
        i64x2(self.0.max(i64x2::splat(0).0))
    }
    fn relu6(self) -> Self {
        i64x2(self.relu().0.min(i64x2::splat(6).0))
    }
}