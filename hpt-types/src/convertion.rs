use half::bf16;
use half::f16;
use hpt_macros::impl_scalar_convert;
use num_complex::{Complex32, Complex64};

use crate::dtype::TypeCommon;
#[cfg(all(
    any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"),
    not(target_feature = "avx2")
))]
use crate::simd::_128bit::*;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
use crate::simd::_256bit::*;

#[cfg(target_feature = "avx512f")]
use crate::simd::_512bit::*;

/// Convertor trait
///
/// This trait is used to convert a scalar to another type
pub(crate) trait Convertor {
    /// convert the value to bool
    fn to_bool(self) -> bool;
    /// convert the value to u8
    fn to_u8(self) -> u8;
    /// convert the value to u16
    fn to_u16(self) -> u16;
    /// convert the value to u32
    fn to_u32(self) -> u32;
    /// convert the value to u64
    fn to_u64(self) -> u64;
    /// convert the value to usize
    fn to_usize(self) -> usize;
    /// convert the value to i8
    fn to_i8(self) -> i8;
    /// convert the value to i16
    fn to_i16(self) -> i16;
    /// convert the value to i32
    fn to_i32(self) -> i32;
    /// convert the value to i64
    fn to_i64(self) -> i64;
    /// convert the value to isize
    fn to_isize(self) -> isize;
    /// convert the value to f32
    fn to_f32(self) -> f32;
    /// convert the value to f64
    fn to_f64(self) -> f64;
    /// convert the value to f16
    fn to_f16(self) -> f16;
    /// convert the value to bf16
    fn to_bf16(self) -> bf16;
    /// convert the value to complex32
    fn to_complex32(self) -> Complex32;
    /// convert the value to complex64
    fn to_complex64(self) -> Complex64;
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub(crate) trait VecConvertor: Sized {
    /// convert the value to boolx32
    fn to_bool(self) -> boolx32 {
        unreachable!()
    }
    /// convert the value to u8x32
    fn to_u8(self) -> u8x32 {
        unreachable!()
    }
    /// convert the value to u16x16
    fn to_u16(self) -> u16x16 {
        unreachable!()
    }
    /// convert the value to u32x8
    fn to_u32(self) -> u32x8 {
        unreachable!()
    }
    /// convert the value to u64x4
    fn to_u64(self) -> u64x4 {
        unreachable!()
    }
    /// convert the value to usizex4
    fn to_usize(self) -> usizex4 {
        unreachable!()
    }
    /// convert the value to i8x32
    fn to_i8(self) -> i8x32 {
        unreachable!()
    }
    /// convert the value to i16x16
    fn to_i16(self) -> i16x16 {
        unreachable!()
    }
    /// convert the value to i32x8
    fn to_i32(self) -> i32x8 {
        unreachable!()
    }
    /// convert the value to i64x4
    fn to_i64(self) -> i64x4 {
        unreachable!()
    }
    /// convert the value to isizex4
    fn to_isize(self) -> isizex4 {
        unreachable!()
    }
    /// convert the value to f32x8
    fn to_f32(self) -> f32x8 {
        unreachable!()
    }
    /// convert the value to f64x4
    fn to_f64(self) -> f64x4 {
        unreachable!()
    }
    /// convert the value to f16x16
    fn to_f16(self) -> f16x16 {
        unreachable!()
    }
    /// convert the value to bf16x16
    fn to_bf16(self) -> bf16x16 {
        unreachable!()
    }
    /// convert the value to cplx32x4
    fn to_complex32(self) -> cplx32x4 {
        unreachable!()
    }
    /// convert the value to cplx64x2
    fn to_complex64(self) -> cplx64x2 {
        unreachable!()
    }
}

#[cfg(target_feature = "avx512f")]
pub(crate) trait VecConvertor: Sized {
    fn to_bool(self) -> boolx64 {
        unreachable!()
    }
    fn to_u8(self) -> u8x64 {
        unreachable!()
    }
    fn to_u16(self) -> u16x32 {
        unreachable!()
    }
    fn to_u32(self) -> u32x16 {
        unreachable!()
    }
    fn to_u64(self) -> u64x8 {
        unreachable!()
    }
    fn to_usize(self) -> usizex8 {
        unreachable!()
    }
    fn to_i8(self) -> i8x64 {
        unreachable!()
    }
    fn to_i16(self) -> i16x32 {
        unreachable!()
    }
    fn to_i32(self) -> i32x16 {
        unreachable!()
    }
    fn to_i64(self) -> i64x8 {
        unreachable!()
    }
    fn to_isize(self) -> isizex8 {
        unreachable!()
    }
    fn to_f32(self) -> f32x16 {
        unreachable!()
    }
    fn to_f64(self) -> f64x8 {
        unreachable!()
    }
    fn to_f16(self) -> f16x32 {
        unreachable!()
    }
    fn to_bf16(self) -> bf16x32 {
        unreachable!()
    }
    fn to_complex32(self) -> cplx32x8 {
        unreachable!()
    }
    fn to_complex64(self) -> cplx64x4 {
        unreachable!()
    }
}


#[cfg(all(
    any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"),
    not(target_feature = "avx2")
))]
/// VecConvertor trait
///
/// This trait is used to convert a simd vector to another type
pub(crate) trait VecConvertor: Sized {
    /// convert the value to boolx16
    fn to_bool(self) -> boolx16 {
        unreachable!()
    }
    /// convert the value to u8x16
    fn to_u8(self) -> u8x16 {
        unreachable!()
    }
    /// convert the value to u16x8
    fn to_u16(self) -> u16x8 {
        unreachable!()
    }
    /// convert the value to u32x4
    fn to_u32(self) -> u32x4 {
        unreachable!()
    }
    /// convert the value to u64x2
    fn to_u64(self) -> u64x2 {
        unreachable!()
    }
    /// convert the value to usizex2
    fn to_usize(self) -> usizex2 {
        unreachable!()
    }
    /// convert the value to i8x16
    fn to_i8(self) -> i8x16 {
        unreachable!()
    }
    /// convert the value to i16x8
    fn to_i16(self) -> i16x8 {
        unreachable!()
    }
    /// convert the value to i32x4
    fn to_i32(self) -> i32x4 {
        unreachable!()
    }
    /// convert the value to i64x2
    fn to_i64(self) -> i64x2 {
        unreachable!()
    }
    /// convert the value to isizex2
    fn to_isize(self) -> isizex2 {
        unreachable!()
    }
    /// convert the value to f32x4
    fn to_f32(self) -> f32x4 {
        unreachable!()
    }
    /// convert the value to f64x2
    fn to_f64(self) -> f64x2 {
        unreachable!()
    }
    /// convert the value to f16x8
    fn to_f16(self) -> f16x8 {
        unreachable!()
    }
    /// convert the value to bf16x8
    fn to_bf16(self) -> bf16x8 {
        unreachable!()
    }
    /// convert the value to cplx32x2
    fn to_complex32(self) -> cplx32x2 {
        unreachable!()
    }
    /// convert the value to cplx64x1
    fn to_complex64(self) -> cplx64x1 {
        unreachable!()
    }
}

impl_scalar_convert!();
