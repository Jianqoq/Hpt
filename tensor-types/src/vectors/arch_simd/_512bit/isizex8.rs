use std::ops::{Deref, DerefMut};

use crate::into_vec::IntoVec;

use crate::vectors::traits::{Init, VecCommon, VecTrait};

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct isizex8(pub(crate) std::simd::isizex8);

impl Deref for isizex8 {
    #[cfg(target_pointer_width = "64")]
    type Target = std::simd::isizex8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for isizex8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl VecTrait<isize> for isizex8 {
    const SIZE: usize = 8;
    type Base = isize;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        self.as_mut_array()
            .copy_from_slice(unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const isize {
        self.as_array().as_ptr() as *const _
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
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
        ret as isize
    }
    fn extract(self, idx: usize) -> isize {
        self.as_array()[idx]
    }
    #[inline(always)]
    fn splat(val: isize) -> Self {
        isizex8(std::simd::isizex8::splat(val))
    }
}

impl VecCommon for isizex8 {
    const SIZE: usize = 8;

    type Base = isize;
}

impl Init<isize> for isizex8 {
    fn splat(val: isize) -> isizex8 {
        let ret = isizex8(std::simd::isizex8::splat(val));
        ret
    }
}

impl std::ops::Add for isizex8 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        isizex8(self.0 + rhs.0)
    }
}
impl std::ops::Sub for isizex8 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        isizex8(self.0 - rhs.0)
    }
}
impl std::ops::Mul for isizex8 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        isizex8(self.0 * rhs.0)
    }
}
impl std::ops::Div for isizex8 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        isizex8(self.0 / rhs.0)
    }
}
impl std::ops::Rem for isizex8 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        isizex8(self.0 % rhs.0)
    }
}
