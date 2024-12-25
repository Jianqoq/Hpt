use crate::convertion::VecConvertor;
use crate::traits::VecTrait;
use crate::vectors::arch_simd::_128bit::u8x16::u8x16;
use crate::traits::SimdCompare;

use super::i8x16::i8x16;

/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct boolx16(pub(crate) [bool; 16]);

impl VecTrait<bool> for boolx16 {
    const SIZE: usize = 16;
    type Base = bool;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[bool]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn sum(&self) -> bool {
        self.0.iter().map(|&x| x as u8).sum::<u8>() > 0
    }
    #[inline(always)]
    fn splat(val: bool) -> boolx16 {
        boolx16([val; 16])
    }
}

impl boolx16 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [bool; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for boolx16 {
    type SimdMask = i8x16;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> i8x16 {
        let mut res = [0i8; 16];
        for i in 0..16 {
            res[i] = if self.0[i] == rhs.0[i] { -1 } else { 0 };
        }
        i8x16(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> i8x16 {
        let mut res = [0i8; 16];
        for i in 0..16 {
            res[i] = if self.0[i] != rhs.0[i] { -1 } else { 0 };
        }
        i8x16(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> i8x16 {
        let mut res = [0i8; 16];
        for i in 0..16 {
            res[i] = if self.0[i] < rhs.0[i] { -1 } else { 0 };
        }
        i8x16(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> i8x16 {
        let mut res = [0i8; 16];
        for i in 0..16 {
            res[i] = if self.0[i] <= rhs.0[i] { -1 } else { 0 };
        }
        i8x16(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> i8x16 {
        let mut res = [0i8; 16];
        for i in 0..16 {
            res[i] = if self.0[i] > rhs.0[i] { -1 } else { 0 };
        }
        i8x16(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> i8x16 {
        let mut res = [0i8; 16];
        for i in 0..16 {
            res[i] = if self.0[i] >= rhs.0[i] { -1 } else { 0 };
        }
        i8x16(unsafe { std::mem::transmute(res) })
    }
}

impl std::ops::Add for boolx16 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        let mask: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(mask | rhs) })
    }
}
impl std::ops::BitAnd for boolx16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        let mask: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(mask & rhs) })
    }
}

impl VecConvertor for boolx16 {
    #[inline(always)]
    fn to_bool(self) -> boolx16 {
        self
    }
    #[inline(always)]
    fn to_i8(self) -> i8x16 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_u8(self) -> u8x16 {
        unsafe { std::mem::transmute(self) }
    }
}
