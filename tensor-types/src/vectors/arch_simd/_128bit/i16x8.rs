use std::ops::{ Deref, DerefMut, Index, IndexMut };

use crate::traits::{Init, SimdSelect, VecTrait};

/// a vector of 8 i16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
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
    fn extract(self, idx: usize) -> i16 {
        self.as_array()[idx]
    }
}

impl SimdSelect<i16x8> for crate::vectors::_128bit::u32x4::u32x4 {
    fn select(&self, true_val: i16x8, false_val: i16x8) -> i16x8 {
        let mask: std::simd::mask16x8 = unsafe { std::mem::transmute(*self) };
        i16x8(mask.select(true_val.0, false_val.0))
    }
}
impl Init<i16> for i16x8 {
    fn splat(val: i16) -> i16x8 {
        i16x8(std::simd::i16x8::splat(val))
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