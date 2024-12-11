use std::ops::{ Deref, DerefMut };

use crate::traits::{Init, SimdSelect, VecTrait};

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
}
impl Init<u64> for u64x2 {
    fn splat(val: u64) -> u64x2 {
        u64x2(std::simd::u64x2::splat(val))
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