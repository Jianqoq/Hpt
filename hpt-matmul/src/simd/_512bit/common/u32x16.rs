
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i32x16::i32x16;

/// a vector of 4 u32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct u32x16(#[cfg(target_arch = "x86_64")] pub(crate) __m512i);


// impl u32x16 {
//     #[inline(always)]
//     pub(crate) fn simd_ne(self, other: Self) -> i32x16 {
//         unsafe {
//             let lhs: i32x16 = std::mem::transmute(self.0);
//             let rhs: i32x16 = std::mem::transmute(other.0);
//             lhs.simd_ne(rhs)
//         }
//     }
//     #[inline(always)]
//     fn simd_le(self, other: Self) -> i32x16 {
//         unsafe {
//             let lhs: i32x16 = std::mem::transmute(self.0);
//             let rhs: i32x16 = std::mem::transmute(other.0);
//             lhs.simd_le(rhs)
//         }
//     }
// 
//     #[inline(always)]
//     pub(crate) fn simd_gt(self, other: Self) -> i32x16 {
//         unsafe {
//             let lhs: i32x16 = std::mem::transmute(self.0);
//             let rhs: i32x16 = std::mem::transmute(other.0);
//             lhs.simd_gt(rhs)
//         }
//     }
// 
//     #[inline(always)]
//     fn simd_ge(self, other: Self) -> i32x16 {
//         unsafe {
//             let lhs: i32x16 = std::mem::transmute(self.0);
//             let rhs: i32x16 = std::mem::transmute(other.0);
//             lhs.simd_ge(rhs)
//         }
//     }
// }