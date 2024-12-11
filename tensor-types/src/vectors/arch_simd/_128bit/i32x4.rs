use std::ops::{ Deref, DerefMut };

use crate::traits::{ Init, SimdSelect, VecTrait };

/// a vector of 4 i32 values
#[allow(non_camel_case_types)]
#[repr(transparent)]
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
}
impl Init<i32> for i32x4 {
    fn splat(val: i32) -> i32x4 {
        i32x4(std::simd::i32x4::splat(val))
    }
}
impl SimdSelect<i32x4> for crate::vectors::_128bit::u32x4::u32x4 {
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