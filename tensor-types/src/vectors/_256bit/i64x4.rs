use std::ops::{ Deref, DerefMut };

use crate::vectors::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct i64x4(pub(crate) std::simd::i64x4);

impl Deref for i64x4 {
    type Target = std::simd::i64x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i64x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i64> for i64x4 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i64]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const i64 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut i64 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut i64 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> i64 {
        self.as_array().iter().sum()
    }
    
    fn extract(self, idx: usize) -> i64 {
        self.as_array()[idx]
    }
}
impl VecSize for i64x4 {
    const SIZE: usize = 4;
}
impl Init<i64> for i64x4 {
    fn splat(val: i64) -> i64x4 {
        i64x4(std::simd::i64x4::splat(val))
    }
}
impl std::ops::Add for i64x4 {
    type Output = i64x4;
    fn add(self, rhs: Self) -> Self::Output {
        i64x4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i64x4 {
    type Output = i64x4;
    fn sub(self, rhs: Self) -> Self::Output {
        i64x4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i64x4 {
    type Output = i64x4;
    fn mul(self, rhs: Self) -> Self::Output {
        i64x4(self.0 * rhs.0)
    }
}
impl std::ops::Div for i64x4 {
    type Output = i64x4;
    fn div(self, rhs: Self) -> Self::Output {
        i64x4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i64x4 {
    type Output = i64x4;
    fn rem(self, rhs: Self) -> Self::Output {
        i64x4(self.0 % rhs.0)
    }
}
