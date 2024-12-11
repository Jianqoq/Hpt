use crate::traits::{Init, VecTrait};
use crate::vectors::arch_simd::_128bit::u8x16::u8x16;
use crate::traits::SimdCompare;

/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
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
}
impl Init<bool> for boolx16 {
    fn splat(val: bool) -> boolx16 {
        boolx16([val; 16])
    }
}
impl SimdCompare for boolx16 {
    type SimdMask = Self;
    fn simd_eq(self, rhs: Self) -> Self {
        let lhs: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(lhs.simd_eq(rhs)) })
    }
    fn simd_ne(self, rhs: Self) -> Self {
        let lhs: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(lhs.simd_ne(rhs)) })
    }
    fn simd_lt(self, rhs: Self) -> Self {
        let lhs: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(lhs.simd_lt(rhs)) })
    }
    fn simd_le(self, rhs: Self) -> Self {
        let lhs: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(lhs.simd_le(rhs)) })
    }
    fn simd_gt(self, rhs: Self) -> Self {
        let lhs: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(lhs.simd_gt(rhs)) })
    }
    fn simd_ge(self, rhs: Self) -> Self {
        let lhs: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(lhs.simd_ge(rhs)) })
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
        let mask: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(mask | rhs) })
    }
}
impl std::ops::BitAnd for boolx16 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mask: u8x16 = unsafe { std::mem::transmute(self) };
        let rhs: u8x16 = unsafe { std::mem::transmute(rhs) };
        boolx16(unsafe { std::mem::transmute(mask & rhs) })
    }
}
