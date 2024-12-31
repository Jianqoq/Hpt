use crate::convertion::VecConvertor;
use crate::traits::SimdCompare;
use crate::traits::{SimdSelect, VecTrait};
use crate::type_promote::{Eval, Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::vectors::arch_simd::_256bit::u8x32::u8x32;

use super::i8x32::i8x32;

/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct boolx32(pub(crate) [bool; 32]);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type bool_promote = boolx32;

impl VecTrait<bool> for boolx32 {
    const SIZE: usize = 32;
    type Base = bool;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[bool]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let mut ret = boolx32::default();
        for i in 0..32 {
            ret.0[i] = (self.0[i] && a.0[i]) || b.0[i];
        }
        ret
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
            ret.0[i] = if arr[i] != 0 {
                true_val.0[i]
            } else {
                false_val.0[i]
            };
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

impl FloatOutBinary2 for boolx32 {
    #[inline(always)]
    fn __div(self, _: Self) -> Self {
        panic!("Division operation is not supported for boolean type")
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for bool")
    }
}

impl NormalOut2 for boolx32 {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn __sub(self, _: Self) -> Self {
        panic!("Subtraction is not supported for boolean type")
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn __pow(self, _: Self) -> Self {
        panic!("Power operation is not supported for boolean type")
    }

    #[inline(always)]
    fn __rem(self, _: Self) -> Self {
        panic!("Remainder operation is not supported for boolean type")
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        self | rhs
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        self & rhs
    }

    #[inline(always)]
    fn __clamp(self, _: Self, _: Self) -> Self {
        self
    }
}

impl NormalOutUnary2 for boolx32 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        self
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        self
    }

    #[inline(always)]
    fn __leaky_relu(self, _: Self) -> Self {
        self
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self
    }
}

impl Eval2 for boolx32 {
    type Output = i8x32;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        unsafe { std::mem::transmute([0i8; 32]) }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            std::mem::transmute([
                self[0]._is_true(),
                self[1]._is_true(),
                self[2]._is_true(),
                self[3]._is_true(),
                self[4]._is_true(),
                self[5]._is_true(),
                self[6]._is_true(),
                self[7]._is_true(),
                self[8]._is_true(),
                self[9]._is_true(),
                self[10]._is_true(),
                self[11]._is_true(),
                self[12]._is_true(),
                self[13]._is_true(),
                self[14]._is_true(),
                self[15]._is_true(),
                self[16]._is_true(),
                self[17]._is_true(),
                self[18]._is_true(),
                self[19]._is_true(),
                self[20]._is_true(),
                self[21]._is_true(),
                self[22]._is_true(),
                self[23]._is_true(),
                self[24]._is_true(),
                self[25]._is_true(),
                self[26]._is_true(),
                self[27]._is_true(),
                self[28]._is_true(),
                self[29]._is_true(),
                self[30]._is_true(),
                self[31]._is_true(),
            ])
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        unsafe {
            std::mem::transmute([
                0i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ])
        }
    }
}
