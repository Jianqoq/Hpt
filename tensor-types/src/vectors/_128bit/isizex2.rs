use std::ops::{ Deref, DerefMut };

use crate::{into_vec::IntoVec, traits::{Init, VecCommon, VecTrait}};

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct isizex2(pub(crate) std::simd::isizex2);

impl Deref for isizex2 {
    type Target = std::simd::isizex2;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for isizex2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl VecTrait<isize> for isizex2 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        self.as_mut_array().copy_from_slice(unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const isize {
        self.as_array().as_ptr() as *const _
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut isize {
        self.as_mut_array().as_mut_ptr() as *mut _
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut isize {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> isize {
        let ret = self.as_array().iter().sum::<isize>();
        ret
    }
    
    fn extract(self, idx: usize) -> isize {
        self.as_array()[idx]
    }
}

impl VecCommon for isizex2 {
    const SIZE: usize = 4;
    
    type Base = isize;
}

impl Init<isize> for isizex2 {
    fn splat(val: isize) -> isizex2 {
        let ret = isizex2(std::simd::isizex2::splat(val));
        ret
    }
    unsafe fn from_ptr(ptr: *const isize) -> Self where Self: Sized {
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

impl IntoVec<isizex2> for isizex2 {
    fn into_vec(self) -> isizex2 {
        self
    }
}
impl std::ops::Add for isizex2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        isizex2(self.0 + rhs.0)
    }
}
impl std::ops::Sub for isizex2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        isizex2(self.0 - rhs.0)
    }
}
impl std::ops::Mul for isizex2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        isizex2(self.0 * rhs.0)
    }
}
impl std::ops::Div for isizex2 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        isizex2(self.0 / rhs.0)
    }
}
impl std::ops::Rem for isizex2 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        isizex2(self.0 % rhs.0)
    }
}