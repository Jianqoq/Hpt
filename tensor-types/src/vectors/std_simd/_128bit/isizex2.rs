use std::{ops::{ Deref, DerefMut }, simd::cmp::{SimdPartialEq, SimdPartialOrd}};

use crate::{impl_std_simd_bit_logic, std_simd::_128bit::i64x2::i64x2, traits::{SimdCompare, SimdMath, VecTrait}};

/// a vector of 2 isize values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
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
    const SIZE: usize = 2;
    type Base = isize;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        self.as_mut_array().copy_from_slice(unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn sum(&self) -> isize {
        let ret = self.as_array().iter().sum::<isize>();
        ret
    }
    fn splat(val: isize) -> isizex2 {
        let ret = isizex2(std::simd::isizex2::splat(val));
        ret
    }
}

impl SimdCompare for isizex2 {
    type SimdMask = isizex2;
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        isizex2(self.0.simd_eq(rhs.0).to_int())
    }
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        isizex2(self.0.simd_ne(rhs.0).to_int())
    }
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        isizex2(self.0.simd_lt(rhs.0).to_int())
    }
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        isizex2(self.0.simd_le(rhs.0).to_int())
    }
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        isizex2(self.0.simd_gt(rhs.0).to_int())
    }
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        isizex2(self.0.simd_ge(rhs.0).to_int())
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
impl std::ops::Neg for isizex2 {
    type Output = Self;
    fn neg(self) -> Self {
        isizex2(-self.0)
    }
}
impl_std_simd_bit_logic!(isizex2);

impl SimdMath<isize> for isizex2 {
    fn max(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                isizex2(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.max(rhs);
                isizex2(std::mem::transmute(ret))
            }
        }
    }
    fn min(self, other: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x2 = std::mem::transmute(self.0);
                let rhs: i64x2 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                isizex2(std::mem::transmute(ret))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                let rhs: i32x4 = std::mem::transmute(other.0);
                let ret = lhs.min(rhs);
                isizex2(std::mem::transmute(ret))
            }
        }
    }
    fn relu(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x2 = std::mem::transmute(self.0);
                isizex2(std::mem::transmute(lhs.relu()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                isizex2(std::mem::transmute(lhs.relu()))
            }
        }
    }
    fn relu6(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            unsafe {
                let lhs: i64x2 = std::mem::transmute(self.0);
                isizex2(std::mem::transmute(lhs.relu6()))
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            unsafe {
                let lhs: i32x4 = std::mem::transmute(self.0);
                isizex2(std::mem::transmute(lhs.relu6()))
            }
        }
    }
}
