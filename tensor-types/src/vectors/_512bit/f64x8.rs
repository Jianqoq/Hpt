use std::{ ops::{ Deref, DerefMut }, simd::StdFloat };

use crate::into_vec::IntoVec;

use crate::vectors::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct f64x8(pub(crate) std::simd::f64x8);

impl Deref for f64x8 {
    type Target = std::simd::f64x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for f64x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<f64> for f64x8 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f64]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const f64 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f64 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut f64 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        self.as_array().iter().sum()
    }

    fn extract(self, idx: usize) -> f64 {
        self.as_array()[idx]
    }
}
impl VecSize for f64x8 {
    const SIZE: usize = 8;
}
impl Init<f64> for f64x8 {
    fn splat(val: f64) -> f64x8 {
        f64x8(std::simd::f64x8::splat(val))
    }

    unsafe fn from_ptr(ptr: *const f64) -> Self {
        unsafe {
            std::mem::transmute(std::simd::f64x8::from_slice(std::slice::from_raw_parts(ptr, 8)))
        }
    }
}
impl IntoVec<f64x8> for f64x8 {
    fn into_vec(self) -> f64x8 {
        self
    }
}

impl std::ops::Add for f64x8 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        f64x8(self.0 + rhs.0)
    }
}
impl std::ops::Sub for f64x8 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        f64x8(self.0 - rhs.0)
    }
}
impl std::ops::Mul for f64x8 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        f64x8(self.0 * rhs.0)
    }
}
impl std::ops::Div for f64x8 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        f64x8(self.0 / rhs.0)
    }
}
impl std::ops::Rem for f64x8 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        f64x8(self.0 % rhs.0)
    }
}
