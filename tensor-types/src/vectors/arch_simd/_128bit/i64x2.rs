use std::ops::{ Deref, DerefMut };

use crate::traits::{ Init, SimdSelect, VecTrait };

/// a vector of 2 i64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
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
}

impl SimdSelect<i64x2> for crate::vectors::_128bit::u64x2::u64x2 {
    fn select(&self, true_val: i64x2, false_val: i64x2) -> i64x2 {
        let mask: std::simd::mask64x2 = unsafe { std::mem::transmute(*self) };
        i64x2(mask.select(true_val.0, false_val.0))
    }
}
impl Init<i64> for i64x2 {
    fn splat(val: i64) -> i64x2 {
        i64x2(std::simd::i64x2::splat(val))
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