use crate::convertion::VecConvertor;
use crate::traits::SimdCompare;
use crate::traits::{SimdSelect, VecTrait};
use crate::type_promote::{Eval, Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::vectors::arch_simd::_512bit::u8x64;

use super::i8x64::i8x64;

/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C, align(64))]
pub struct boolx64(pub(crate) [bool; 64]);

impl Default for boolx64 {
    fn default() -> Self {
        Self([false; 64])
    }
}

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type bool_promote = boolx64;

impl VecTrait<bool> for boolx64 {
    const SIZE: usize = 64;
    type Base = bool;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let mut ret = boolx64::default();
        for i in 0..64 {
            ret.0[i] = (self.0[i] && a.0[i]) || b.0[i];
        }
        ret
    }
    #[inline(always)]
    fn sum(&self) -> bool {
        self.0.iter().map(|&x| x as u8).sum::<u8>() > 0
    }
    #[inline(always)]
    fn splat(val: bool) -> boolx64 {
        boolx64([val; 64])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const bool) -> Self {
        boolx64([
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
            ptr.add(31 + 1).read_unaligned(),
            ptr.add(31 + 2).read_unaligned(),
            ptr.add(31 + 3).read_unaligned(),
            ptr.add(31 + 4).read_unaligned(),
            ptr.add(31 + 5).read_unaligned(),
            ptr.add(31 + 6).read_unaligned(),
            ptr.add(31 + 7).read_unaligned(),
            ptr.add(31 + 8).read_unaligned(),
            ptr.add(31 + 9).read_unaligned(),
            ptr.add(31 + 10).read_unaligned(),
            ptr.add(31 + 11).read_unaligned(),
            ptr.add(31 + 12).read_unaligned(),
            ptr.add(31 + 13).read_unaligned(),
            ptr.add(31 + 14).read_unaligned(),
            ptr.add(31 + 15).read_unaligned(),
            ptr.add(31 + 16).read_unaligned(),
            ptr.add(31 + 17).read_unaligned(),
            ptr.add(31 + 18).read_unaligned(),
            ptr.add(31 + 19).read_unaligned(),
            ptr.add(31 + 20).read_unaligned(),
            ptr.add(31 + 21).read_unaligned(),
            ptr.add(31 + 22).read_unaligned(),
            ptr.add(31 + 23).read_unaligned(),
            ptr.add(31 + 24).read_unaligned(),
            ptr.add(31 + 25).read_unaligned(),
            ptr.add(31 + 26).read_unaligned(),
            ptr.add(31 + 27).read_unaligned(),
            ptr.add(31 + 28).read_unaligned(),
            ptr.add(31 + 29).read_unaligned(),
            ptr.add(31 + 30).read_unaligned(),
            ptr.add(31 + 31).read_unaligned(),
            ptr.add(31 + 32).read_unaligned(), 
        ])
    }
}


impl SimdCompare for boolx64 {
    type SimdMask = i8x64;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> i8x64 {
        let mut res = [0i8; 64];
        for i in 0..64 {
            res[i] = if self.0[i] == rhs.0[i] { -1 } else { 0 };
        }
        i8x64(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> i8x64 {
        let mut res = [0i8; 64];
        for i in 0..64 {
            res[i] = if self.0[i] != rhs.0[i] { -1 } else { 0 };
        }
        i8x64(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> i8x64 {
        let mut res = [0i8; 64];
        for i in 0..64 {
            res[i] = if !self.0[i] & rhs.0[i] { -1 } else { 0 };
        }
        i8x64(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> i8x64 {
        let mut res = [0i8; 64];
        for i in 0..64 {
            res[i] = if self.0[i] <= rhs.0[i] { -1 } else { 0 };
        }
        i8x64(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> i8x64 {
        let mut res = [0i8; 64];
        for i in 0..64 {
            res[i] = if self.0[i] & !rhs.0[i] { -1 } else { 0 };
        }
        i8x64(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> i8x64 {
        let mut res = [0i8; 64];
        for i in 0..64 {
            res[i] = if self.0[i] >= rhs.0[i] { -1 } else { 0 };
        }
        i8x64(unsafe { std::mem::transmute(res) })
    }
}

impl SimdSelect<boolx64> for i8x64 {
    #[inline(always)]
    fn select(&self, true_val: boolx64, false_val: boolx64) -> boolx64 {
        let mut ret = boolx64::default();
        let arr: [bool; 64] = unsafe { std::mem::transmute(self.0) };
        for i in 0..64 {
            ret.0[i] = if arr[i] {
                true_val.0[i]
            } else {
                false_val.0[i]
            };
        }
        ret
    }
}

impl std::ops::Add for boolx64 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = boolx64::default();
        for i in 0..64 {
            ret.0[i] = self.0[i] || rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for boolx64 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = boolx64::default();
        for i in 0..64 {
            ret.0[i] = self.0[i] && !rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for boolx64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = boolx64::default();
        for i in 0..64 {
            ret.0[i] = self.0[i] && rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for boolx64 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = boolx64::default();
        for i in 0..64 {
            ret.0[i] = self.0[i] && !rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for boolx64 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = boolx64::default();
        for i in 0..64 {
            ret.0[i] = self.0[i] ^ rhs.0[i];
        }
        ret
    }
}
impl std::ops::BitOr for boolx64 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        let mask: u8x64 = unsafe { std::mem::transmute(self) };
        let rhs: u8x64 = unsafe { std::mem::transmute(rhs) };
        boolx64(unsafe { std::mem::transmute(mask | rhs) })
    }
}
impl std::ops::BitAnd for boolx64 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        let mask: u8x64 = unsafe { std::mem::transmute(self) };
        let rhs: u8x64 = unsafe { std::mem::transmute(rhs) };
        boolx64(unsafe { std::mem::transmute(mask & rhs) })
    }
}

impl VecConvertor for boolx64 {
    #[inline(always)]
    fn to_bool(self) -> boolx64 {
        self
    }
    #[inline(always)]
    fn to_i8(self) -> i8x64 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_u8(self) -> u8x64 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for boolx64 {
    #[inline(always)]
    fn __div(self, _: Self) -> Self {
        panic!("Division operation is not supported for boolean type")
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for bool")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for boolx64");
    }

    #[inline(always)]
    fn __pow(self, _: Self) -> Self {
        panic!("Power operation is not supported for boolean type")
    }
}

impl NormalOut2 for boolx64 {
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

impl NormalOutUnary2 for boolx64 {
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

    #[inline(always)]
    fn __trunc(self) -> Self {
        self
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        let mut ret = boolx64::default();
        for i in 0..64 {
            ret.0[i] = self.0[i] ^ rhs.0[i];
        }
        ret
    }
}

impl Eval2 for boolx64 {
    type Output = i8x64;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        unsafe { std::mem::transmute([0i8; 64]) }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        let mut ret = [0i8; 64];
        for i in 0..64 {
            ret[i] = self.0[i].__is_true() as i8;
        }
        unsafe { std::mem::transmute(ret) }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        unsafe {
            std::mem::transmute([
                0i8; 64
            ])
        }
    }
}
