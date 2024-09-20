use std::ops::{ Deref, DerefMut, Index, IndexMut };

use crate::vectors::traits::{ Init, VecCommon, VecTrait };

/// a vector of 8 i32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct i32x8(pub(crate) std::simd::i32x8);

impl Deref for i32x8 {
    type Target = std::simd::i32x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i32x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i32> for i32x8 {
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i32]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const i32 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut i32 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut i32 {
        self.as_array().as_ptr() as *mut _
    }
    fn extract(self, idx: usize) -> i32 {
        self.as_array()[idx]
    }

    #[inline(always)]
    fn sum(&self) -> i32 {
        self.as_array().iter().sum()
    }
}
impl VecCommon for i32x8 {
    const SIZE: usize = 8;
    
    type Base = i32;
}
impl Init<i32> for i32x8 {
    fn splat(val: i32) -> i32x8 {
        i32x8(std::simd::i32x8::splat(val))
    }
}
impl Index<usize> for i32x8 {
    type Output = i32;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for i32x8 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl std::ops::Add  for i32x8 {
    type Output = i32x8;
    fn add(self, rhs: Self) -> Self::Output {
        i32x8(self.0 + rhs.0)
    }
}
impl std::ops::Sub  for i32x8 {
    type Output = i32x8;
    fn sub(self, rhs: Self) -> Self::Output {
        i32x8(self.0 - rhs.0)
    }
}
impl std::ops::Mul  for i32x8 {
    type Output = i32x8;
    fn mul(self, rhs: Self) -> Self::Output {
        i32x8(self.0 * rhs.0)
    }
}
impl std::ops::Div  for i32x8 {
    type Output = i32x8;
    fn div(self, rhs: Self) -> Self::Output {
        i32x8(self.0 / rhs.0)
    }
}
impl std::ops::Rem  for i32x8 {
    type Output = i32x8;
    fn rem(self, rhs: Self) -> Self::Output {
        i32x8(self.0 % rhs.0)
    }
}