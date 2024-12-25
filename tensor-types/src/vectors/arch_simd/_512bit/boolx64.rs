use std::simd::{ cmp::SimdPartialEq, Simd };
use std::simd::cmp::SimdPartialOrd;
use crate::into_vec::IntoVec;

use crate::vectors::traits::{ Init, VecCommon, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct boolx64(pub(crate) [bool; 64]);

impl VecTrait<bool> for boolx64 {
    const SIZE: usize = 64;
    type Base = bool;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[bool]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const bool {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn mul_add(self, _: Self, _: Self) -> Self {
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
        self.0
            .iter()
            .map(|&x| x as u8)
            .sum::<u8>() > 0
    }
    #[inline(always)]
    fn splat(val: bool) -> Self {
        boolx64([val; 64])
    }
}
impl VecCommon for boolx64 {
    const SIZE: usize = 64;
    
    type Base = bool;
}
impl Init<bool> for boolx64 {
    fn splat(val: bool) -> boolx64 {
        boolx64([val; 64])
    }
}

impl boolx64 {
    pub fn simd_eq(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 64> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 64> = unsafe { std::mem::transmute(rhs) };
        boolx64(lhs.simd_eq(rhs).into())
    }
    pub fn simd_ne(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 64> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 64> = unsafe { std::mem::transmute(rhs) };
        boolx64(lhs.simd_ne(rhs).into())
    }
    pub fn simd_lt(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 64> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 64> = unsafe { std::mem::transmute(rhs) };
        boolx64(lhs.simd_lt(rhs).into())
    }
    pub fn simd_le(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 64> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 64> = unsafe { std::mem::transmute(rhs) };
        boolx64(lhs.simd_le(rhs).into())
    }
    pub fn simd_gt(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 64> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 64> = unsafe { std::mem::transmute(rhs) };
        boolx64(lhs.simd_gt(rhs).into())
    }
    pub fn simd_ge(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 64> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 64> = unsafe { std::mem::transmute(rhs) };
        boolx64(lhs.simd_ge(rhs).into())
    }
}

impl std::ops::Add for boolx64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = [false; 64];
        for i in 0..64 {
            ret[i] = self.0[i] || rhs.0[i];
        }
        boolx64(ret)
    }
}
impl std::ops::Sub for boolx64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = [false; 64];
        for i in 0..64 {
            ret[i] = self.0[i] && !rhs.0[i];
        }
        boolx64(ret)
    }
}
impl std::ops::Mul for boolx64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = [false; 64];
        for i in 0..64 {
            ret[i] = self.0[i] && rhs.0[i];
        }
        boolx64(ret)
    }
}
impl std::ops::Div for boolx64 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = [false; 64];
        for i in 0..64 {
            ret[i] = self.0[i] && !rhs.0[i];
        }
        boolx64(ret)
    }
}
impl std::ops::Rem for boolx64 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = [false; 64];
        for i in 0..64 {
            ret[i] = self.0[i] ^ rhs.0[i];
        }
        boolx64(ret)
    }
}
impl std::ops::BitOr for boolx64 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mask: Simd<u8, 64> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 64> = unsafe { std::mem::transmute(rhs) };
        boolx64(unsafe { std::mem::transmute(mask | rhs) })
    }
}
impl std::ops::BitAnd for boolx64 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mask: Simd<u8, 64> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 64> = unsafe { std::mem::transmute(rhs) };
        boolx64(unsafe { std::mem::transmute(mask & rhs) })
    }
}
