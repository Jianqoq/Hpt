use crate::convertion::VecConvertor;
use crate::traits::{SimdSelect, VecTrait};
use crate::vectors::arch_simd::_256bit::u8x32::u8x32;
use crate::traits::SimdCompare;

use super::i8x32::i8x32;

/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct boolx32(pub(crate) [bool; 32]);

impl VecTrait<bool> for boolx32 {
    const SIZE: usize = 32;
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
    fn splat(val: bool) -> boolx32 {
        boolx32([val; 32])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const bool) -> Self {
        boolx32([
            ptr.read_unaligned(),
            ptr.add(1).read_unaligned(),
            ptr.add(2).read_unaligned(),
            ptr.add(3).read_unaligned(),
            ptr.add(4).read_unaligned(),
            ptr.add(5).read_unaligned(),
            ptr.add(6).read_unaligned(),
            ptr.add(7).read_unaligned(),
            ptr.add(8).read_unaligned(),
            ptr.add(9).read_unaligned(),
            ptr.add(10).read_unaligned(),
            ptr.add(11).read_unaligned(),
            ptr.add(12).read_unaligned(),
            ptr.add(13).read_unaligned(),
            ptr.add(14).read_unaligned(),
            ptr.add(15).read_unaligned(),
            ptr.add(16).read_unaligned(),
            ptr.add(17).read_unaligned(),
            ptr.add(18).read_unaligned(),
            ptr.add(19).read_unaligned(),
            ptr.add(20).read_unaligned(),
            ptr.add(21).read_unaligned(),
            ptr.add(22).read_unaligned(),
            ptr.add(23).read_unaligned(),
            ptr.add(24).read_unaligned(),
            ptr.add(25).read_unaligned(),
            ptr.add(26).read_unaligned(),
            ptr.add(27).read_unaligned(),
            ptr.add(28).read_unaligned(),
            ptr.add(29).read_unaligned(),
            ptr.add(30).read_unaligned(),
            ptr.add(31).read_unaligned(),
        ])
    }
}

impl boolx32 {
    /// convert the vector to an array
    #[inline(always)]
    pub fn as_array(&self) -> [bool; 32] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for boolx32 {
    type SimdMask = i8x32;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> i8x32 {
        let mut res = [0i8; 32];
        for i in 0..32 {
            res[i] = if self.0[i] == rhs.0[i] { -1 } else { 0 };
        }
        i8x32(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> i8x32 {
        let mut res = [0i8; 32];
        for i in 0..32 {
            res[i] = if self.0[i] != rhs.0[i] { -1 } else { 0 };
        }
        i8x32(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> i8x32 {
        let mut res = [0i8; 32];
        for i in 0..32 {
            res[i] = if self.0[i] < rhs.0[i] { -1 } else { 0 };
        }
        i8x32(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> i8x32 {
        let mut res = [0i8; 32];
        for i in 0..32 {
            res[i] = if self.0[i] <= rhs.0[i] { -1 } else { 0 };
        }
        i8x32(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> i8x32 {
        let mut res = [0i8; 32];
        for i in 0..32 {
            res[i] = if self.0[i] > rhs.0[i] { -1 } else { 0 };
        }
        i8x32(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> i8x32 {
        let mut res = [0i8; 32];
        for i in 0..32 {
            res[i] = if self.0[i] >= rhs.0[i] { -1 } else { 0 };
        }
        i8x32(unsafe { std::mem::transmute(res) })
    }
}

impl SimdSelect<boolx32> for i8x32 {
    #[inline(always)]
    fn select(&self, true_val: boolx32, false_val: boolx32) -> boolx32 {
        let mut ret = boolx32::default();
        let arr = self.as_array();
        for i in 0..32 {
            ret.0[i] = if arr[i] != 0 { true_val.0[i] } else { false_val.0[i] };
        }
        ret
    }
}

impl std::ops::Add for boolx32 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        let mask: u8x32 = unsafe { std::mem::transmute(self) };
        let rhs: u8x32 = unsafe { std::mem::transmute(rhs) };
        boolx32(unsafe { std::mem::transmute(mask | rhs) })
    }
}
impl std::ops::BitAnd for boolx32 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        let mask: u8x32 = unsafe { std::mem::transmute(self) };
        let rhs: u8x32 = unsafe { std::mem::transmute(rhs) };
        boolx32(unsafe { std::mem::transmute(mask & rhs) })
    }
}

impl VecConvertor for boolx32 {
    #[inline(always)]
    fn to_bool(self) -> boolx32 {
        self
    }
    #[inline(always)]
    fn to_i8(self) -> i8x32 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_u8(self) -> u8x32 {
        unsafe { std::mem::transmute(self) }
    }
}
