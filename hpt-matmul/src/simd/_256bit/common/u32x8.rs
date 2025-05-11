
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i32x8::i32x8;

/// a vector of 4 u32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u32x8(#[cfg(target_arch = "x86_64")] pub(crate) __m256i);


impl u32x8 {
    #[inline(always)]
    pub(crate) fn simd_ne(self, other: Self) -> i32x8 {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i32x8 {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }

    #[inline(always)]
    pub(crate) fn simd_gt(self, other: Self) -> i32x8 {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }

    #[inline(always)]
    fn simd_ge(self, other: Self) -> i32x8 {
        unsafe {
            let lhs: i32x8 = std::mem::transmute(self.0);
            let rhs: i32x8 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}