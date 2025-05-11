
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::simd::_256bit::common::i16x16::i16x16;

/// a vector of 16 u16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u16x16(#[cfg(target_arch = "x86_64")] pub(crate) __m256i);

impl u16x16 {
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i16x16 {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i16x16 {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }
    #[inline(always)]
    pub(crate) fn simd_gt(self, other: Self) -> i16x16 {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i16x16 {
        unsafe {
            let lhs: i16x16 = std::mem::transmute(self.0);
            let rhs: i16x16 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}