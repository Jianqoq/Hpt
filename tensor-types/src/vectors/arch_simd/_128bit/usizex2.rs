use crate::{
    arch_simd::_128bit::u64x2::u64x2,
    convertion::VecConvertor,
    traits::{ SimdCompare, SimdMath, VecTrait },
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{ i64x2::i64x2, isizex2::isizex2 };

/// a vector of 2 usize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
#[cfg(target_pointer_width = "64")]
pub struct usizex2(pub(crate) u64x2);
#[cfg(target_pointer_width = "32")]
pub struct usizex4(pub(crate) u32x4);

#[cfg(target_pointer_width = "32")]
type USizeVEC = usizex4;
#[cfg(target_pointer_width = "64")]
type USizeVEC = usizex2;

#[cfg(target_pointer_width = "32")]
type USizeBase = u32x4;
#[cfg(target_pointer_width = "64")]
type USizeBase = u64x2;

impl PartialEq for usizex2 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Default for usizex2 {
    #[inline(always)]
    fn default() -> Self {
        Self(USizeBase::default())
    }
}

impl VecTrait<usize> for usizex2 {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 2;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 4;
    type Base = usize;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[usize]) {
        USizeBase::copy_from_slice(&mut self.0, unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }
    #[inline(always)]
    fn sum(&self) -> usize {
        self.0.sum() as usize
    }
    #[inline(always)]
    fn splat(val: usize) -> Self {
        Self(USizeBase::splat(val as u64))
    }
}

impl USizeVEC {
    #[cfg(target_pointer_width = "64")]
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [usize; 2] {
        unsafe { std::mem::transmute(self.0) }
    }
    #[cfg(target_pointer_width = "32")]
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [usize; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for USizeVEC {
    #[cfg(target_pointer_width = "64")]
    type SimdMask = isizex2;
    #[cfg(target_pointer_width = "32")]
    type SimdMask = isizex4;

    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(self.0.simd_eq(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(self.0.simd_eq(other.0))
        }
    }

    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(self.0.simd_ne(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(self.0.simd_ne(other.0))
        }
    }

    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(self.0.simd_lt(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(self.0.simd_lt(other.0))
        }
    }

    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(self.0.simd_le(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(self.0.simd_le(other.0))
        }
    }

    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(self.0.simd_gt(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(self.0.simd_gt(other.0))
        }
    }

    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex2(self.0.simd_ge(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex4(self.0.simd_ge(other.0))
        }
    }
}

impl std::ops::Add for usizex2 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}
impl std::ops::Sub for usizex2 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}
impl std::ops::Mul for usizex2 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}
impl std::ops::Div for usizex2 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.div(rhs.0))
    }
}
impl std::ops::Rem for usizex2 {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0.rem(rhs.0))
    }
}
impl std::ops::BitAnd for usizex2 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0.bitand(rhs.0))
    }
}
impl std::ops::BitOr for usizex2 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0.bitor(rhs.0))
    }
}
impl std::ops::BitXor for usizex2 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0.bitxor(rhs.0))
    }
}
impl std::ops::Not for usizex2 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(self.0.not())
    }
}
impl std::ops::Shl for usizex2 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        Self(self.0.shl(rhs.0))
    }
}
impl std::ops::Shr for usizex2 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        Self(self.0.shr(rhs.0))
    }
}
impl SimdMath<usize> for usizex2 {
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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

impl VecConvertor for usizex2 {
    #[inline(always)]
    fn to_usize(self) -> usizex2 {
        self
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "64")]
    fn to_u64(self) -> u64x2 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_u32(self) -> u32x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_f32(self) -> super::f32x4::f32x4 {
        self.to_u32().to_f32()
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "64")]
    fn to_i64(self) -> i64x2 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_i32(self) -> i32x4 {
        unsafe { std::mem::transmute(self) }
    }
}
