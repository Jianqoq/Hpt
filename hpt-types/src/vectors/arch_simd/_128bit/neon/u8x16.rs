use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::vectors::arch_simd::_128bit::boolx16;
use crate::vectors::arch_simd::_128bit::i8x16;
use crate::vectors::arch_simd::_128bit::u8x16;

impl PartialEq for u8x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = vceqq_u8(self.0, other.0);
            vmaxvq_u8(cmp) == 0xff && vminvq_u8(cmp) == 0xff
        }
    }
}

impl Default for u8x16 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { u8x16(vdupq_n_u8(0)) }
    }
}

impl VecTrait<u8> for u8x16 {
    const SIZE: usize = 16;
    type Base = u8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u8]) {
        unsafe {
            self.0 = vld1q_u8(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_u8(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> u8 {
        unsafe {
            let x: [u8; 16] = std::mem::transmute(self.0);
            x.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: u8) -> u8x16 {
        unsafe { u8x16(vdupq_n_u8(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const u8) -> Self {
        unsafe { u8x16(vld1q_u8(ptr)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
}

impl std::ops::Add for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { u8x16(vaddq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { u8x16(vsubq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { u8x16(vmulq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 16] = std::mem::transmute(self.0);
            let arr2: [u8; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 16] = [0; 16];
            for i in 0..16 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            return u8x16(vld1q_u8(arr3.as_ptr()));
        }
    }
}
impl std::ops::Rem for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 16] = std::mem::transmute(self.0);
            let arr2: [u8; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] % arr2[i];
            }
            return u8x16(vld1q_u8(arr3.as_ptr()));
        }
    }
}

impl std::ops::BitAnd for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { u8x16(vandq_u8(self.0, rhs.0)) }
    }
}

impl std::ops::BitOr for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { u8x16(vorrq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { u8x16(veorq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { u8x16(vmvnq_u8(self.0)) }
    }
}
impl std::ops::Shl for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        unsafe { u8x16(vshlq_u8(self.0, vreinterpretq_s8_u8(rhs.0))) }
    }
}
impl std::ops::Shr for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        unsafe {
            let a: [u8; 16] = std::mem::transmute(self.0);
            let b: [u8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            return u8x16(vld1q_u8(result.as_ptr()));
        }
    }
}

impl SimdMath<u8> for u8x16 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { u8x16(vmaxq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { u8x16(vminq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    fn relu(self) -> Self {
        self.max(Self::splat(0))
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        self.min(Self::splat(6)).max(Self::splat(0))
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        self
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        self
    }
    #[inline(always)]
    fn round(self) -> Self {
        self
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self
    }
    #[inline(always)]
    fn pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [u8; 16] = std::mem::transmute(self.0);
            let b: [u8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0u8; 16];
            for i in 0..16 {
                result[i] = a[i].pow(b[i] as u32);
            }
            return u8x16(vld1q_u8(result.as_ptr()));
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl FloatOutBinary2 for u8x16 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for u8")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for u8x16");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 16] = std::mem::transmute(self.0);
            let arr2: [u8; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            return u8x16(vld1q_u8(arr3.as_ptr()));
        }
    }
}

impl Eval2 for u8x16 {
    type Output = i8x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i8x16::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            i8x16(vmvnq_s8(vreinterpretq_s8_u8(vceqq_u8(
                self.0,
                vdupq_n_u8(0),
            ))))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i8x16::default()
    }
}
