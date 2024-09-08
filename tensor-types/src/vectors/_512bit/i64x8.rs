use std::ops::{ Deref, DerefMut };

use crate::vectors::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct i64x8(pub(crate) std::simd::i64x8);

impl Deref for i64x8 {
    type Target = std::simd::i64x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i64x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i64> for i64x8 {
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
impl VecSize for i64x8 {
    const SIZE: usize = 8;
}
impl Init<i64> for i64x8 {
    fn splat(val: i64) -> i64x8 {
        i64x8(std::simd::i64x8::splat(val))
    }

    unsafe fn from_ptr(ptr: *const i64) -> Self {
        unsafe { std::mem::transmute(
            std::simd::i64x8::from_slice(std::slice::from_raw_parts(ptr, 8))
        ) }
    }
}
impl std::ops::Add for i64x8 {
    type Output = i64x8;
    fn add(self, rhs: Self) -> Self::Output {
        i64x8(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i64x8 {
    type Output = i64x8;
    fn sub(self, rhs: Self) -> Self::Output {
        i64x8(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i64x8 {
    type Output = i64x8;
    fn mul(self, rhs: Self) -> Self::Output {
        i64x8(self.0 * rhs.0)
    }
}
impl std::ops::Div for i64x8 {
    type Output = i64x8;
    fn div(self, rhs: Self) -> Self::Output {
        i64x8(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i64x8 {
    type Output = i64x8;
    fn rem(self, rhs: Self) -> Self::Output {
        i64x8(self.0 % rhs.0)
    }
}
