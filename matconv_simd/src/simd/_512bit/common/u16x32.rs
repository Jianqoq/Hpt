
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 16 u16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct u16x32(#[cfg(target_arch = "x86_64")] pub(crate) __m256i);

// impl u16x32 {
//     #[inline(always)]
//     fn simd_lt(self, other: Self) -> i16x32 {
//         unsafe {
//             let lhs: i16x32 = std::mem::transmute(self.0);
//             let rhs: i16x32 = std::mem::transmute(other.0);
//             lhs.simd_lt(rhs)
//         }
//     }
//     #[inline(always)]
//     fn simd_le(self, other: Self) -> i16x32 {
//         unsafe {
//             let lhs: i16x32 = std::mem::transmute(self.0);
//             let rhs: i16x32 = std::mem::transmute(other.0);
//             lhs.simd_le(rhs)
//         }
//     }
//     #[inline(always)]
//     pub(crate) fn simd_gt(self, other: Self) -> i16x32 {
//         unsafe {
//             let lhs: i16x32 = std::mem::transmute(self.0);
//             let rhs: i16x32 = std::mem::transmute(other.0);
//             lhs.simd_gt(rhs)
//         }
//     }
//     #[inline(always)]
//     fn simd_ge(self, other: Self) -> i16x32 {
//         unsafe {
//             let lhs: i16x32 = std::mem::transmute(self.0);
//             let rhs: i16x32 = std::mem::transmute(other.0);
//             lhs.simd_ge(rhs)
//         }
//     }
// }