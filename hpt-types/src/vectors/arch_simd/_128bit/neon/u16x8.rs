use crate::{
    traits::{SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2},
};

use std::arch::aarch64::*;

use crate::vectors::arch_simd::_128bit::i16x8;
use crate::vectors::arch_simd::_128bit::u16x8;

impl PartialEq for u16x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = vceqq_u16(self.0, other.0);
            vaddvq_u16(cmp) == 8
        }
    }
}

impl Default for u16x8 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { u16x8(vdupq_n_u16(0)) }
    }
}

impl VecTrait<u16> for u16x8 {
    const SIZE: usize = 8;
    type Base = u16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u16]) {
        unsafe {
            self.0 = vld1q_u16(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_u16(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        unsafe { vaddvq_u16(self.0) }
    }
    #[inline(always)]
    fn splat(val: u16) -> u16x8 {
        unsafe { u16x8(vdupq_n_u16(val)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const u16) -> Self {
        unsafe { u16x8(vld1q_u16(ptr)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_laneq_u16::<LANE>(b.0, self.0, a.0)) }
    }
}

impl SimdSelect<u16x8> for u16x8 {
    #[inline(always)]
    fn select(&self, true_val: u16x8, false_val: u16x8) -> u16x8 {
        unsafe { u16x8(vbslq_u16(self.0, true_val.0, false_val.0)) }
    }
}

impl std::ops::Add for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(vaddq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(vsubq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(vmulq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            return u16x8(vld1q_u16(arr3.as_ptr()));
        }
    }
}
impl std::ops::Rem for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            return u16x8(vld1q_u16(arr3.as_ptr()));
        }
    }
}

impl std::ops::BitAnd for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(vandq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(vorrq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u16x8 {
    type Output = u16x8;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(veorq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { u16x8(vmvnq_u16(self.0)) }
    }
}
impl std::ops::Shl for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe { u16x8(vshlq_u16(self.0, vreinterpretq_s16_u16(rhs.0))) }
    }
}
impl std::ops::Shr for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 8] = std::mem::transmute(self.0);
            let b: [u16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            return u16x8(vld1q_u16(result.as_ptr()));
        }
    }
}

impl SimdMath<u16> for u16x8 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { u16x8(vmaxq_u16(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { u16x8(vminq_u16(self.0, other.0)) }
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
            let a: [u16; 8] = std::mem::transmute(self.0);
            let b: [u16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0u16; 8];
            for i in 0..8 {
                result[i] = a[i].pow(b[i] as u32);
            }
            return u16x8(vld1q_u16(result.as_ptr()));
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl FloatOutBinary2 for u16x8 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for u16")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for u16x8");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            return u16x8(vld1q_u16(arr3.as_ptr()));
        }
    }
}

impl Eval2 for u16x8 {
    type Output = i16x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i16x8::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            i16x8(vmvnq_s16(vreinterpretq_s16_u16(vceqq_u16(
                self.0,
                vdupq_n_u16(0),
            ))))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i16x8::default()
    }
}
