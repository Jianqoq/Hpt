use std::ops::{ Deref, DerefMut, Index, IndexMut };

use crate::traits::{ Init, SimdSelect, VecTrait };

/// a vector of 4 u32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
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
    fn extract(self, idx: usize) -> u32 {
        self.as_array()[idx]
    }
}

impl SimdSelect<u32x4> for u32x4 {
    fn select(&self, true_val: u32x4, false_val: u32x4) -> u32x4 {
        let mask: std::simd::mask32x4 = unsafe { std::mem::transmute(*self) };
        u32x4(mask.select(true_val.0, false_val.0))
    }
}
impl Init<u32> for u32x4 {
    fn splat(val: u32) -> u32x4 {
        u32x4(std::simd::u32x4::splat(val))
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
