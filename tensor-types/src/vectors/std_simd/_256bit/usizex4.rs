use std::ops::{ Deref, DerefMut, Index, IndexMut };

use crate::vectors::traits::{ Init, VecTrait };

/// a vector of 4 usize values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct usizex4(pub(crate) std::simd::usizex4);

impl Deref for usizex4 {
    type Target = std::simd::usizex4;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for usizex4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl VecTrait<usize> for usizex4 {
    const SIZE: usize = 4;
    type Base = usize;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[usize]) {
        self.as_mut_array().copy_from_slice(unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const usize {
        self.as_array().as_ptr() as *const _
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
}

impl Init<usize> for usizex4 {
    fn splat(val: usize) -> usizex4 {
        #[cfg(target_pointer_width = "64")]
        let ret = usizex4(std::simd::usizex4::splat(val));
        #[cfg(target_pointer_width = "32")]
        let ret = usizex8(std::simd::usizex8::splat(val));
        ret
    }
}
impl Index<usize> for usizex4 {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}
impl IndexMut<usize> for usizex4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}
impl std::ops::Add for usizex4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        usizex4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for usizex4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        usizex4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for usizex4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        usizex4(self.0 * rhs.0)
    }
}
impl std::ops::Div for usizex4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        usizex4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for usizex4 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        usizex4(self.0 % rhs.0)
    }
}
