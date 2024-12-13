use std::{ops::{ Deref, DerefMut }, simd::{cmp::{SimdPartialEq, SimdPartialOrd}, num::SimdUint, Simd}};

use crate::{impl_std_simd_bit_logic, std_simd::_256bit::u64x4::u64x4, traits::{SimdCompare, SimdMath}, vectors::traits::VecTrait};

use super::isizex4::isizex4;

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
    fn sum(&self) -> usize {
        self.as_array().iter().sum::<usize>()
    }
    fn splat(val: usize) -> usizex4 {
        #[cfg(target_pointer_width = "64")]
        let ret = usizex4(std::simd::usizex4::splat(val));
        #[cfg(target_pointer_width = "32")]
        let ret = usizex8(std::simd::usizex8::splat(val));
        ret
    }
}

impl SimdCompare for usizex4 {
    type SimdMask = isizex4;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<usize, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<usize, 4> = unsafe { std::mem::transmute(rhs) };
        isizex4(lhs.simd_eq(rhs).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<usize, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<usize, 4> = unsafe { std::mem::transmute(rhs) };
        isizex4(lhs.simd_ne(rhs).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<usize, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<usize, 4> = unsafe { std::mem::transmute(rhs) };
        isizex4(lhs.simd_lt(rhs).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<usize, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<usize, 4> = unsafe { std::mem::transmute(rhs) };
        isizex4(lhs.simd_le(rhs).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<usize, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<usize, 4> = unsafe { std::mem::transmute(rhs) };
        isizex4(lhs.simd_gt(rhs).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        let lhs: Simd<usize, 4> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<usize, 4> = unsafe { std::mem::transmute(rhs) };
        isizex4(lhs.simd_ge(rhs).to_int())
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

impl_std_simd_bit_logic!(usizex4);

impl SimdMath<usize> for usizex4 {
    fn max(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x4 = std::mem::transmute(self.0);
                let rhs: u64x4 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                usizex4(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x8 = std::mem::transmute(self.0);
                let rhs: u32x8 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                usizex4(std::mem::transmute(ret))
            }
        }
    }
    fn min(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x4 = std::mem::transmute(self.0);
                let rhs: u64x4 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                usizex4(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x8 = std::mem::transmute(self.0);
                let rhs: u32x8 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                usizex4(std::mem::transmute(ret))
            }
        }
    }
    fn relu(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x4 = std::mem::transmute(self.0);
                usizex4(std::mem::transmute(lhs.relu()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x8 = std::mem::transmute(self.0);
                usizex4(std::mem::transmute(lhs.relu()))
            }
        }
    }
    fn relu6(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: u64x4 = std::mem::transmute(self.0);
                usizex4(std::mem::transmute(lhs.relu6()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: u32x8 = std::mem::transmute(self.0);
                usizex4(std::mem::transmute(lhs.relu6()))
            }
        }
    }
    fn neg(self) -> Self {
        usizex4(self.0.wrapping_neg())
    }
}
