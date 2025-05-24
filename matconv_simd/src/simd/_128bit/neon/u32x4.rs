

use std::arch::aarch64::*;

use crate::{simd::_128bit::common::{i32x4::i32x4, u32x4::u32x4}, VecTrait};

impl VecTrait<u32> for u32x4 {
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_u32(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn splat(val: u32) -> u32x4 {
        unsafe { u32x4(vdupq_n_u32(val)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        unsafe { Self(vmlaq_laneq_u32::<LANE>(b.0, self.0, a.0)) }
    }
    #[inline(always)]
    fn partial_load(ptr: *const u32, num_elem: usize) -> Self {
        let mut result = Self::splat(u32::default());
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, (&mut result.0) as *mut _ as *mut u32, num_elem);
            result
        }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut u32, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping((&self.0) as *const _ as *const u32, ptr, num_elem);
        }
    }
}

impl i32x4 {
    #[inline(always)]
    pub(crate) fn select_u32x4(&self, true_val: u32x4, false_val: u32x4) -> u32x4 {
        unsafe {
            u32x4(vbslq_u32(
                vreinterpretq_u32_s32(self.0),
                true_val.0,
                false_val.0,
            ))
        }
    }
}

impl std::ops::Add for u32x4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(vaddq_u32(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u32x4 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(vsubq_u32(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(vmulq_u32(self.0, rhs.0)) }
    }
}

impl std::ops::BitAnd for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(vandq_u32(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(vorrq_u32(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(veorq_u32(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe { u32x4(vmvnq_u32(self.0)) }
    }
}
impl std::ops::Shl for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(vshlq_u32(self.0, vreinterpretq_s32_u32(rhs.0))) }
    }
}
impl std::ops::Shr for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u32; 4] = std::mem::transmute(self.0);
            let b: [u32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            return u32x4(vld1q_u32(result.as_ptr()));
        }
    }
}