use std::ops::{ Deref, DerefMut };

use crate::into_vec::IntoVec;

use crate::vectors::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct usizex8(pub(crate) std::simd::usizex8);

impl Deref for usizex8 {
    type Target = std::simd::usizex8;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for usizex8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl VecTrait<usize> for usizex8 {
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

impl VecSize for usizex8 {
    const SIZE: usize = 8;
}

impl Init<usize> for usizex8 {
    fn splat(val: usize) -> usizex8 {
        let ret = usizex8(std::simd::usizex8::splat(val));
        ret
    }
}

impl IntoVec<usizex8> for usizex8 {
    fn into_vec(self) -> usizex8 {
        self
    }
}
impl std::ops::Add for usizex8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        usizex8(self.0 + rhs.0)
    }
}
impl std::ops::Sub for usizex8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        usizex8(self.0 - rhs.0)
    }
}
impl std::ops::Mul for usizex8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        usizex8(self.0 * rhs.0)
    }
}
impl std::ops::Div for usizex8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        usizex8(self.0 / rhs.0)
    }
}
impl std::ops::Rem for usizex8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        usizex8(self.0 % rhs.0)
    }
}
