use std::ops::{Deref, DerefMut};

use crate::into_vec::IntoVec;

use crate::vectors::traits::{Init, VecCommon, VecTrait};

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct u64x8(pub(crate) std::simd::u64x8);

impl Deref for u64x8 {
    type Target = std::simd::u64x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u64x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u64> for u64x8 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u64]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u64 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u64 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut u64 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> u64 {
        self.as_array().iter().sum()
    }

    fn extract(self, idx: usize) -> u64 {
        self.as_array()[idx]
    }
}
impl VecCommon for u64x8 {
    const SIZE: usize = 8;

    type Base = u64;
}
impl Init<u64> for u64x8 {
    fn splat(val: u64) -> u64x8 {
        u64x8(std::simd::u64x8::splat(val))
    }
}
impl std::ops::Add for u64x8 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        u64x8(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u64x8 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        u64x8(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u64x8 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        u64x8(self.0 * rhs.0)
    }
}
impl std::ops::Div for u64x8 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        u64x8(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u64x8 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        u64x8(self.0 % rhs.0)
    }
}
