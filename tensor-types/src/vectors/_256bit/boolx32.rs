use std::ops::{Index, IndexMut};
use std::simd::cmp::SimdPartialOrd;
use std::simd::{cmp::SimdPartialEq, Simd};

use crate::traits::SimdCompare;
use crate::vectors::traits::{Init, VecCommon, VecTrait};

/// a vector of 32 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct boolx32(pub(crate) [bool; 32]);

impl VecTrait<bool> for boolx32 {
    #[inline(always)]
    fn mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[bool]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const bool {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut bool {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut bool {
        self.0.as_ptr() as *mut _
    }
    fn extract(self, idx: usize) -> bool {
        self.0[idx]
    }

    #[inline(always)]
    fn sum(&self) -> bool {
        self.0.iter().map(|&x| x as u8).sum::<u8>() > 0
    }
}
impl VecCommon for boolx32 {
    const SIZE: usize = 32;

    type Base = bool;
}
impl Init<bool> for boolx32 {
    fn splat(val: bool) -> boolx32 {
        boolx32([val; 32])
    }
}

impl Index<usize> for boolx32 {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for boolx32 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl SimdCompare for boolx32 {
    type SimdMask = Self;
    fn simd_eq(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_eq(rhs).into())
    }
    fn simd_ne(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_ne(rhs).into())
    }
    fn simd_lt(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_lt(rhs).into())
    }
    fn simd_le(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_le(rhs).into())
    }
    fn simd_gt(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_gt(rhs).into())
    }
    fn simd_ge(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_ge(rhs).into())
    }
}

impl std::ops::Add for boolx32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = boolx32::default();
        for i in 0..32 {
            ret.0[i] = self.0[i] || rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for boolx32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = boolx32::default();
        for i in 0..32 {
            ret.0[i] = self.0[i] && !rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for boolx32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = boolx32::default();
        for i in 0..32 {
            ret.0[i] = self.0[i] && rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for boolx32 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = boolx32::default();
        for i in 0..32 {
            ret.0[i] = self.0[i] && !rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for boolx32 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = boolx32::default();
        for i in 0..32 {
            ret.0[i] = self.0[i] ^ rhs.0[i];
        }
        ret
    }
}
impl std::ops::BitOr for boolx32 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mask: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(unsafe { std::mem::transmute(mask | rhs) })
    }
}
impl std::ops::BitAnd for boolx32 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mask: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(unsafe { std::mem::transmute(mask & rhs) })
    }
}
