
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i32x4::i32x4;

/// a vector of 4 u32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u32x4(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")] pub(crate) uint32x4_t,
);


impl u32x4 {
    #[inline(always)]
    pub(crate) fn simd_ne(self, other: Self) -> i32x4 {
        unsafe {
            let lhs: i32x4 = std::mem::transmute(self.0);
            let rhs: i32x4 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }
    #[inline(always)]
    pub(crate) fn simd_gt(self, other: Self) -> i32x4 {
        unsafe {
            let lhs: i32x4 = std::mem::transmute(self.0);
            let rhs: i32x4 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }
}
