use crate::convertion::VecConvertor;
use crate::traits::SimdCompare;
use crate::traits::VecTrait;
use crate::type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::vectors::arch_simd::_128bit::u8x16::u8x16;

use super::i8x16::i8x16;

/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct boolx16(pub(crate) [bool; 16]);

#[allow(non_camel_case_types)]
pub(crate) type bool_promote = boolx16;

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
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const bool) -> Self {
        let mut result = [false; 16];
        for i in 0..16 {
            result[i] = unsafe { *ptr.add(i) };
        }
        boolx16(result)
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

impl FloatOutBinary2 for boolx16 {
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
        panic!("Hypotenuse operation is not supported for bool")
    }

    #[inline(always)]
    fn __pow(self, _: Self) -> Self {
        panic!("Power operation is not supported for boolean type")
    }
}

impl NormalOut2 for boolx16 {
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

impl NormalOutUnary2 for boolx16 {
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
    fn __trunc(self) -> Self {
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

    fn __copysign(self, rhs: Self) -> Self {
        let mut ret = boolx16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] ^ rhs.0[i];
        }
        ret
    }
}

impl Eval2 for boolx16 {
    type Output = i8x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        unsafe { std::mem::transmute(boolx16::default()) }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(boolx16::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        unsafe { std::mem::transmute(boolx16::default()) }
    }
}
