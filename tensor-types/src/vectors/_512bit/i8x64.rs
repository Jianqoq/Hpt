use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use crate::vectors::traits::{ Init, VecCommon, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct i8x64(pub(crate) std::simd::i8x64);

impl Deref for i8x64 {
    type Target = std::simd::i8x64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i8x64 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i8> for i8x64 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i8]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const i8 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut i8 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut i8 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        self.as_array().iter().sum()
    }
    
    fn extract(self, idx: usize) -> i8 {
        self.as_array()[idx]
    }
}
impl VecCommon for i8x64 {
    const SIZE: usize = 64;
    
    type Base = i8;
}
impl Init<i8> for i8x64 {
    fn splat(val: i8) -> i8x64 {
        i8x64(std::simd::i8x64::splat(val))
    }
}
impl std::ops::Add for i8x64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        i8x64(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i8x64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        i8x64(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i8x64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        i8x64(self.0 * rhs.0)
    }
}
impl std::ops::Div for i8x64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        i8x64(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i8x64 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        i8x64(self.0 % rhs.0)
    }
}