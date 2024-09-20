use crate::traits::{Init, VecCommon, VecTrait};
use std::ops::{Index, IndexMut};
use std::simd::cmp::SimdPartialOrd;
use std::simd::{cmp::SimdPartialEq, Simd};

use crate::{traits::SimdCompare, vectors::_128bit::u8x16::u8x16};

/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct boolx16(pub(crate) [bool; 16]);

impl VecTrait<bool> for boolx16 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[bool]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const bool {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut bool {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut bool {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> bool {
        self.0.iter().map(|&x| x as u8).sum::<u8>() > 0
    }

    fn extract(self, idx: usize) -> bool {
        self.0[idx]
    }
}
impl VecCommon for boolx16 {
    const SIZE: usize = 16;

    type Base = bool;
}
impl Init<bool> for boolx16 {
    fn splat(val: bool) -> boolx16 {
        boolx16([val; 16])
    }
}
impl Index<usize> for boolx16 {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for boolx16 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl SimdCompare for boolx16 {
    type SimdMask = Self;
    fn simd_eq(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        boolx16(lhs.simd_eq(rhs).into())
    }
    fn simd_ne(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        boolx16(lhs.simd_ne(rhs).into())
    }
    fn simd_lt(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        boolx16(lhs.simd_lt(rhs).into())
    }
    fn simd_le(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        boolx16(lhs.simd_le(rhs).into())
    }
    fn simd_gt(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        boolx16(lhs.simd_gt(rhs).into())
    }
    fn simd_ge(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        boolx16(lhs.simd_ge(rhs).into())
    }
}

impl std::ops::Add for boolx16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = boolx16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] || rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for boolx16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = boolx16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] && !rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for boolx16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = boolx16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] && rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for boolx16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = boolx16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] && !rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for boolx16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = boolx16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] ^ rhs.0[i];
        }
        ret
    }
}
impl std::ops::BitOr for boolx16 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mask: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(mask | rhs) })
    }
}
impl std::ops::BitAnd for boolx16 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mask: Simd<u8, 16> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 16> = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(mask & rhs) })
    }
}
