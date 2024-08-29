use std::simd::{ cmp::SimdPartialEq, Simd };
use std::simd::cmp::SimdPartialOrd;
use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct boolx32(pub(crate) [bool; 32]);

impl VecTrait<bool> for boolx32 {
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
        self.0
            .iter()
            .map(|&x| x as u8)
            .sum::<u8>() > 0
    }
}
impl VecSize for boolx32 {
    const SIZE: usize = 32;
}
impl Init<bool> for boolx32 {
    fn splat(val: bool) -> boolx32 {
        boolx32([val; 32])
    }

    unsafe fn from_ptr(ptr: *const bool) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_si256(ptr as *const _)) }
    }
}
impl IntoVec<boolx32> for boolx32 {
    fn into_vec(self) -> boolx32 {
        self
    }
}

impl boolx32 {
    pub fn simd_eq(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_eq(rhs).into())
    }
    pub fn simd_ne(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_ne(rhs).into())
    }
    pub fn simd_lt(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_lt(rhs).into())
    }
    pub fn simd_le(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_le(rhs).into())
    }
    pub fn simd_gt(self, rhs: Self) -> Self {
        let lhs: Simd<u8, 32> = unsafe { std::mem::transmute(self) };
        let rhs: Simd<u8, 32> = unsafe { std::mem::transmute(rhs) };
        boolx32(lhs.simd_gt(rhs).into())
    }
    pub fn simd_ge(self, rhs: Self) -> Self {
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
