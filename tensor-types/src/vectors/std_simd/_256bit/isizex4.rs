use std::ops::{ Deref, DerefMut };

use crate::vectors::traits::VecTrait;

/// a vector of 4 isize values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct isizex4(pub(crate) std::simd::isizex4);

impl Deref for isizex4 {
    #[cfg(target_pointer_width = "64")]
    type Target = std::simd::isizex4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for isizex4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl VecTrait<isize> for isizex4 {
    const SIZE: usize = 4;
    type Base = isize;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        self.as_mut_array().copy_from_slice(unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn sum(&self) -> isize {
        let ret = self.as_array().iter().sum::<isize>();
        ret
    }
    fn splat(val: isize) -> isizex4 {
        isizex4(std::simd::isizex4::splat(val))
    }
}

impl std::ops::Add for isizex4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        isizex4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for isizex4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        isizex4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for isizex4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        isizex4(self.0 * rhs.0)
    }
}
impl std::ops::Div for isizex4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        isizex4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for isizex4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        isizex4(self.0 % rhs.0)
    }
}
impl std::ops::Neg for isizex4 {
    type Output = Self;
    fn neg(self) -> Self {
        isizex4(-self.0)
    }
}