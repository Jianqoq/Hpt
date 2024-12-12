use std::{ops::{ Deref, DerefMut }, simd::cmp::{SimdPartialEq, SimdPartialOrd}};

use crate::{impl_std_simd_bit_logic, traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait }};

/// a vector of 4 i32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct i32x4(pub(crate) std::simd::i32x4);

impl Deref for i32x4 {
    type Target = std::simd::i32x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i32x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i32> for i32x4 {
    const SIZE: usize = 4;
    type Base = i32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i32]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn sum(&self) -> i32 {
        self.as_array().iter().sum()
    }
    fn splat(val: i32) -> i32x4 {
        i32x4(std::simd::i32x4::splat(val))
    }
}

impl SimdCompare for i32x4 {
    type SimdMask = i32x4;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_eq(rhs.0).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_ne(rhs.0).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_lt(rhs.0).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_le(rhs.0).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_gt(rhs.0).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        i32x4(self.0.simd_ge(rhs.0).to_int())
    }
}

impl SimdSelect<i32x4> for crate::vectors::std_simd::_128bit::u32x4::u32x4 {
    fn select(&self, true_val: i32x4, false_val: i32x4) -> i32x4 {
        let mask: std::simd::mask32x4 = unsafe { std::mem::transmute(*self) };
        i32x4(mask.select(true_val.0, false_val.0))
    }
}

impl std::ops::Add for i32x4 {
    type Output = i32x4;
    fn add(self, rhs: Self) -> Self::Output {
        i32x4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i32x4 {
    type Output = i32x4;
    fn sub(self, rhs: Self) -> Self::Output {
        i32x4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i32x4 {
    type Output = i32x4;
    fn mul(self, rhs: Self) -> Self::Output {
        i32x4(self.0 * rhs.0)
    }
}
impl std::ops::Div for i32x4 {
    type Output = i32x4;
    fn div(self, rhs: Self) -> Self::Output {
        i32x4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i32x4 {
    type Output = i32x4;
    fn rem(self, rhs: Self) -> Self::Output {
        i32x4(self.0 % rhs.0)
    }
}
impl std::ops::Neg for i32x4 {
    type Output = i32x4;
    fn neg(self) -> Self::Output {
        i32x4(-self.0)
    }
}
impl_std_simd_bit_logic!(i32x4);

impl SimdMath<i32> for i32x4 {
    fn max(self, other: Self) -> Self {
        i32x4(self.0.max(other.0))
    }
    fn min(self, other: Self) -> Self {
        i32x4(self.0.min(other.0))
    }
    fn relu(self) -> Self {
        i32x4(self.0.max(i32x4::splat(0).0))
    }
    fn relu6(self) -> Self {
        i32x4(self.relu().0.min(i32x4::splat(6).0))
    }
}