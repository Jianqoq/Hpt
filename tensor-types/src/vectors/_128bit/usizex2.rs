use std::ops::{ Deref, DerefMut };

use crate::{ into_vec::IntoVec, traits::{ Init, VecCommon, VecTrait } };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct usizex2(pub(crate) std::simd::usizex2);

impl Deref for usizex2 {
    type Target = std::simd::usizex2;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for usizex2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl VecTrait<usize> for usizex2 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[usize]) {
        self.as_mut_array().copy_from_slice(unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const usize {
        self.as_array().as_ptr() as *const _
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut usize {
        self.as_mut_array().as_mut_ptr() as *mut _
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut usize {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> usize {
        self.as_array().iter().sum::<usize>()
    }

    fn extract(self, idx: usize) -> usize {
        self.as_array()[idx]
    }
}

impl VecCommon for usizex2 {
    const SIZE: usize = 2;
    
    type Base = usize;
}

impl Init<usize> for usizex2 {
    fn splat(val: usize) -> usizex2 {
        let ret = usizex2(std::simd::usizex2::splat(val));
        ret
    }
    unsafe fn from_ptr(ptr: *const usize) -> Self where Self: Sized {
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

impl IntoVec<usizex2> for usizex2 {
    fn into_vec(self) -> usizex2 {
        self
    }
}
impl std::ops::Add for usizex2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        usizex2(self.0 + rhs.0)
    }
}
impl std::ops::Sub for usizex2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        usizex2(self.0 - rhs.0)
    }
}
impl std::ops::Mul for usizex2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        usizex2(self.0 * rhs.0)
    }
}
impl std::ops::Div for usizex2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        usizex2(self.0 / rhs.0)
    }
}
impl std::ops::Rem for usizex2 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        usizex2(self.0 % rhs.0)
    }
}
