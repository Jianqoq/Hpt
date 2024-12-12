use std::ops::{ Deref, DerefMut };

use crate::{arch_simd::_128bit::u64x2::u64x2, traits::{SimdMath, VecTrait}};

/// a vector of 2 usize values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
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
    const SIZE: usize = 2;
    type Base = usize;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[usize]) {
        self.as_mut_array().copy_from_slice(unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn sum(&self) -> usize {
        self.as_array().iter().sum::<usize>()
    }
    fn splat(val: usize) -> usizex2 {
        usizex2(std::simd::usizex2::splat(val))
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

impl SimdMath<usize> for usizex2 {
    fn max(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x2 = std::mem::transmute(self.0);
                let rhs: u64x2 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                usizex2(std::mem::transmute(ret.0))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x4 = std::mem::transmute(self.0);
                let rhs: u32x4 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                usizex2(std::mem::transmute(ret.0))
            }
        }
    }
    fn min(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x2 = std::mem::transmute(self.0);
                let rhs: u64x2 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                usizex2(std::mem::transmute(ret.0))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x4 = std::mem::transmute(self.0);
                let rhs: u32x4 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                usizex2(std::mem::transmute(ret.0))
            }
        }
    }
    fn relu(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x2 = std::mem::transmute(self.0);
                usizex2(std::mem::transmute(lhs.relu()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x4 = std::mem::transmute(self.0);
                usizex2(std::mem::transmute(lhs.relu()))
            }
        }
    }
    fn relu6(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x2 = std::mem::transmute(self.0);
                usizex2(std::mem::transmute(lhs.relu6()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x4 = std::mem::transmute(self.0);
                usizex2(std::mem::transmute(lhs.relu6()))
            }
        }
    }
}
