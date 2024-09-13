use std::ops::{ Deref, DerefMut };

use crate::traits::{ Init, VecCommon, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
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
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i32]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const i32 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut i32 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut i32 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> i32 {
        self.as_array().iter().sum()
    }

    fn extract(self, idx: usize) -> i32 {
        self.as_array()[idx]
    }
}
impl VecCommon for i32x4 {
    const SIZE: usize = 4;

    type Base = i32;
}
impl Init<i32> for i32x4 {
    fn splat(val: i32) -> i32x4 {
        i32x4(std::simd::i32x4::splat(val))
    }
    unsafe fn from_ptr(ptr: *const i32) -> Self where Self: Sized {
        #[cfg(target_feature = "neon")]
        {
            unsafe { std::mem::transmute(std::arch::aarch64::_simd_loadu_si128(ptr as *const _)) }
        }
        #[cfg(not(target_feature = "neon"))]
        {
            unsafe { std::mem::transmute(std::arch::x86_64::_mm_loadu_si128(ptr as *const _)) }
        }
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
