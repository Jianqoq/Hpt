use half::f16;
use half::bf16;
use num_complex::{ Complex64, Complex32 };
use tensor_macros::impl_simd_convert;
use std::simd::num::SimdFloat;
#[cfg(target_feature = "avx2")]
use crate::vectors::_256bit::*;
#[cfg(all(
    any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"),
    not(target_feature = "avx2")
))]
use crate::vectors::_128bit::*;
#[cfg(target_feature = "avx512f")]
use crate::vectors::_512bit::*;
use crate::dtype::TypeCommon;
use std::simd::num::SimdInt;
use std::simd::num::SimdUint;

/// Convertor trait
/// 
/// This trait is used to convert a scalar to another type
/// 
/// # Example
/// 
/// ```
/// use tensor_types::Convertor;
/// 
/// let a = 1u8;
/// 
/// let b = a.to_f32();
/// 
/// assert_eq!(b, 1.0);
/// ```
pub trait Convertor {
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
    /// convert the value to u128
    fn to_u128(self) -> u128;
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
    /// convert the value to i128
    fn to_i128(self) -> i128;
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

/// VecConvertor trait
/// 
/// This trait is used to convert a simd vector to another type
/// 
/// # Example
/// 
/// ```
/// use tensor_types::VecConvertor;
/// 
/// let a = f32x8::f32x8::splat(1.0);
/// 
/// let b = a.to_f64();
/// 
/// assert_eq!(b, f64x4::f64x4::splat(1.0));
/// ```
#[cfg(target_feature = "avx2")]
pub trait VecConvertor {
    /// convert the value to boolx32
    fn to_bool(self) -> boolx32::boolx32;
    /// convert the value to u8x32
    fn to_u8(self) -> u8x32::u8x32;
    /// convert the value to u16x16
    fn to_u16(self) -> u16x16::u16x16;
    /// convert the value to u32x8
    fn to_u32(self) -> u32x8::u32x8;
    /// convert the value to u64x4
    fn to_u64(self) -> u64x4::u64x4;
    /// convert the value to usizex4
    fn to_usize(self) -> usizex4::usizex4;
    /// convert the value to i8x32
    fn to_i8(self) -> i8x32::i8x32;
    /// convert the value to i16x16
    fn to_i16(self) -> i16x16::i16x16;
    /// convert the value to i32x8
    fn to_i32(self) -> i32x8::i32x8;
    /// convert the value to i64x4
    fn to_i64(self) -> i64x4::i64x4;
    /// convert the value to isizex4
    fn to_isize(self) -> isizex4::isizex4;
    /// convert the value to f32x8
    fn to_f32(self) -> f32x8::f32x8;
    /// convert the value to f64x4
    fn to_f64(self) -> f64x4::f64x4;
    /// convert the value to f16x16
    fn to_f16(self) -> f16x16::f16x16;
    /// convert the value to bf16x16
    fn to_bf16(self) -> bf16x16::bf16x16;
    /// convert the value to cplx32x4
    fn to_complex32(self) -> cplx32x4::cplx32x4;
    /// convert the value to cplx64x2
    fn to_complex64(self) -> cplx64x2::cplx64x2;
}


#[cfg(all(
    any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"),
    not(target_feature = "avx2")
))]
/// VecConvertor trait
/// 
/// This trait is used to convert a simd vector to another type
/// 
/// # Example
/// 
/// ```
/// use tensor_types::VecConvertor;
/// 
/// let a = f32x8::f32x8::splat(1.0);
/// 
/// let b = a.to_f64();
/// 
/// assert_eq!(b, f64x4::f64x4::splat(1.0));
/// ```
pub trait VecConvertor {
    /// convert the value to boolx16
    fn to_bool(self) -> boolx16::boolx16;
    /// convert the value to u8x16
    fn to_u8(self) -> u8x16::u8x16;
    /// convert the value to u16x8
    fn to_u16(self) -> u16x8::u16x8;
    /// convert the value to u32x4
    fn to_u32(self) -> u32x4::u32x4;
    /// convert the value to u64x2
    fn to_u64(self) -> u64x2::u64x2;
    /// convert the value to usizex2
    fn to_usize(self) -> usizex2::usizex2;
    /// convert the value to i8x16
    fn to_i8(self) -> i8x16::i8x16;
    /// convert the value to i16x8
    fn to_i16(self) -> i16x8::i16x8;
    /// convert the value to i32x4
    fn to_i32(self) -> i32x4::i32x4;
    /// convert the value to i64x2
    fn to_i64(self) -> i64x2::i64x2;
    /// convert the value to isizex2
    fn to_isize(self) -> isizex2::isizex2;
    /// convert the value to f32x4
    fn to_f32(self) -> f32x4::f32x4;
    /// convert the value to f64x2
    fn to_f64(self) -> f64x2::f64x2;
    /// convert the value to f16x8
    fn to_f16(self) -> f16x8::f16x8;
    /// convert the value to bf16x8
    fn to_bf16(self) -> bf16x8::bf16x8;
    /// convert the value to cplx32x2
    fn to_complex32(self) -> cplx32x2::cplx32x2;
    /// convert the value to cplx64x1
    fn to_complex64(self) -> cplx64x1::cplx64x1;
}

impl_simd_convert!();

impl Convertor for bool {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as u8 as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as u8 as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as u8 as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as u8 as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as u8 as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as u8 as f64, 0.0)
    }
}

impl Convertor for u8 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for u16 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for u32 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for u64 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for u128 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for usize {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for i8 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for i16 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for i32 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for i64 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for i128 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }
    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for isize {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        if self < 0 { 0 } else { self as u8 }
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        if self < 0 { 0 } else { self as u16 }
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        if self < 0 { 0 } else { self as u32 }
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        if self < 0 { 0 } else { self as u64 }
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        if self < 0 { 0 } else { self as u128 }
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        if self < 0 { 0 } else { self as usize }
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }

    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }

    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for f32 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0.0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self)
    }

    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self)
    }

    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl Convertor for f64 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self != 0.0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self as f32)
    }

    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self as f32)
    }

    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Convertor for f16 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self.to_f32() != 0.0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self.to_f32() as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self.to_f32() as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self.to_f32() as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self.to_f32() as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self.to_f32() as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self.to_f32() as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self.to_f32() as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self.to_f32() as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self.to_f32() as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self.to_f32() as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self.to_f32() as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self.to_f32() as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    #[inline(always)]
    fn to_f16(self) -> f16 {
        self
    }

    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self.to_f32())
    }

    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self.to_f32(), 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self.to_f64(), 0.0)
    }
}

impl Convertor for Complex32 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self.re as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self.re as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self.re as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self.re as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self.re as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self.re as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self.re as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self.re as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self.re as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self.re as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self.re as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self.re as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.re
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.re as f64
    }

    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self.re)
    }

    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self.re)
    }

    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        self
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }
}

impl Convertor for Complex64 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self.re as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self.re as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self.re as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self.re as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self.re as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self.re as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self.re as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self.re as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self.re as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self.re as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self.re as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self.re as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.re as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.re
    }

    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self.re as f32)
    }

    fn to_bf16(self) -> bf16 {
        bf16::from_f32(self.re as f32)
    }

    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self.re as f32, self.im as f32)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        self
    }
}

impl Convertor for bf16 {
    #[inline(always)]
    fn to_bool(self) -> bool {
        self.to_f32() != 0.0
    }
    #[inline(always)]
    fn to_u8(self) -> u8 {
        self.to_f32() as u8
    }
    #[inline(always)]
    fn to_u16(self) -> u16 {
        self.to_f32() as u16
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self.to_f32() as u32
    }
    #[inline(always)]
    fn to_u64(self) -> u64 {
        self.to_f32() as u64
    }
    #[inline(always)]
    fn to_u128(self) -> u128 {
        self.to_f32() as u128
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self.to_f32() as usize
    }
    #[inline(always)]
    fn to_i8(self) -> i8 {
        self.to_f32() as i8
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self.to_f32() as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self.to_f32() as i32
    }
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self.to_f32() as i64
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self.to_f32() as i128
    }
    #[inline(always)]
    fn to_isize(self) -> isize {
        self.to_f32() as isize
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    #[inline(always)]
    fn to_f16(self) -> f16 {
        f16::from_f32(self.to_f32())
    }

    #[inline(always)]
    fn to_bf16(self) -> bf16 {
        self
    }

    #[inline(always)]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self.to_f32(), 0.0)
    }

    #[inline(always)]
    fn to_complex64(self) -> Complex64 {
        Complex64::new(self.to_f64(), 0.0)
    }
}

/// A trait for converting a scalar value to another scalar value.
pub trait FromScalar<T> {
    /// Convert a scalar value to another scalar value.
    fn __from(a: T) -> Self;
}

impl FromScalar<bool> for bool {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        a
    }
}

impl FromScalar<bool> for u8 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for u16 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for u32 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for u64 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for usize {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for i8 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for i16 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for i32 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for i64 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for isize {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1 } else { 0 }
    }
}

impl FromScalar<bool> for f32 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1.0 } else { 0.0 }
    }
}

impl FromScalar<bool> for f64 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { 1.0 } else { 0.0 }
    }
}

impl FromScalar<bool> for f16 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { f16::ONE } else { f16::ZERO }
    }
}

impl FromScalar<bool> for Complex32 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { Complex32::new(1.0, 0.0) } else { Complex32::new(0.0, 0.0) }
    }
}

impl FromScalar<bool> for Complex64 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) }
    }
}

impl FromScalar<bool> for bf16 {
    #[inline(always)]
    fn __from(a: bool) -> Self {
        if a { bf16::ONE } else { bf16::ZERO }
    }
}

impl FromScalar<u8> for bool {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a != 0
    }
}

impl FromScalar<u8> for u8 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a
    }
}

impl FromScalar<u8> for u16 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as u16
    }
}

impl FromScalar<u8> for u32 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as u32
    }
}

impl FromScalar<u8> for u64 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as u64
    }
}

impl FromScalar<u8> for usize {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as usize
    }
}

impl FromScalar<u8> for i8 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as i8
    }
}

impl FromScalar<u8> for i16 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as i16
    }
}

impl FromScalar<u8> for i32 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as i32
    }
}

impl FromScalar<u8> for i64 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as i64
    }
}

impl FromScalar<u8> for isize {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as isize
    }
}

impl FromScalar<u8> for f32 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as f32
    }
}

impl FromScalar<u8> for f64 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        a as f64
    }
}

impl FromScalar<u8> for f16 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<u8> for Complex32 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<u8> for Complex64 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<u8> for bf16 {
    #[inline(always)]
    fn __from(a: u8) -> Self {
        bf16::from_f32(a as f32)
    }
}

impl FromScalar<u16> for bool {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a != 0
    }
}

impl FromScalar<u16> for u8 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as u8
    }
}

impl FromScalar<u16> for u16 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a
    }
}

impl FromScalar<u16> for u32 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as u32
    }
}

impl FromScalar<u16> for u64 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as u64
    }
}

impl FromScalar<u16> for usize {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as usize
    }
}

impl FromScalar<u16> for i8 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as i8
    }
}

impl FromScalar<u16> for i16 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as i16
    }
}

impl FromScalar<u16> for i32 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as i32
    }
}

impl FromScalar<u16> for i64 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as i64
    }
}

impl FromScalar<u16> for isize {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as isize
    }
}

impl FromScalar<u16> for f32 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as f32
    }
}

impl FromScalar<u16> for f64 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        a as f64
    }
}

impl FromScalar<u16> for f16 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<u16> for Complex32 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<u16> for Complex64 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<u16> for bf16 {
    #[inline(always)]
    fn __from(a: u16) -> Self {
        bf16::from_f32(a as f32)
    }
}

impl FromScalar<u32> for bool {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a != 0
    }
}

impl FromScalar<u32> for u8 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as u8
    }
}

impl FromScalar<u32> for u16 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as u16
    }
}

impl FromScalar<u32> for u32 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a
    }
}

impl FromScalar<u32> for u64 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as u64
    }
}

impl FromScalar<u32> for usize {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as usize
    }
}

impl FromScalar<u32> for i8 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as i8
    }
}

impl FromScalar<u32> for i16 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as i16
    }
}

impl FromScalar<u32> for i32 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as i32
    }
}

impl FromScalar<u32> for i64 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as i64
    }
}

impl FromScalar<u32> for isize {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as isize
    }
}

impl FromScalar<u32> for f32 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as f32
    }
}

impl FromScalar<u32> for f64 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        a as f64
    }
}

impl FromScalar<u32> for f16 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<u32> for Complex32 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<u32> for Complex64 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<u32> for bf16 {
    #[inline(always)]
    fn __from(a: u32) -> Self {
        bf16::from_f64(a as f64)
    }
}

impl FromScalar<u64> for bool {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a != 0
    }
}

impl FromScalar<u64> for u8 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as u8
    }
}

impl FromScalar<u64> for u16 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as u16
    }
}

impl FromScalar<u64> for u32 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as u32
    }
}

impl FromScalar<u64> for u64 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a
    }
}

impl FromScalar<u64> for usize {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as usize
    }
}

impl FromScalar<u64> for i8 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as i8
    }
}

impl FromScalar<u64> for i16 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as i16
    }
}

impl FromScalar<u64> for i32 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as i32
    }
}

impl FromScalar<u64> for i64 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as i64
    }
}

impl FromScalar<u64> for isize {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as isize
    }
}

impl FromScalar<u64> for f32 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as f32
    }
}

impl FromScalar<u64> for f64 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        a as f64
    }
}

impl FromScalar<u64> for f16 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<u64> for Complex32 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<u64> for Complex64 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<u64> for bf16 {
    #[inline(always)]
    fn __from(a: u64) -> Self {
        bf16::from_f64(a as f64)
    }
}

impl FromScalar<usize> for bool {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a != 0
    }
}

impl FromScalar<usize> for u8 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as u8
    }
}

impl FromScalar<usize> for u16 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as u16
    }
}

impl FromScalar<usize> for u32 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as u32
    }
}

impl FromScalar<usize> for u64 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as u64
    }
}

impl FromScalar<usize> for usize {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a
    }
}

impl FromScalar<usize> for i8 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as i8
    }
}

impl FromScalar<usize> for i16 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as i16
    }
}

impl FromScalar<usize> for i32 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as i32
    }
}

impl FromScalar<usize> for i64 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as i64
    }
}

impl FromScalar<usize> for isize {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as isize
    }
}

impl FromScalar<usize> for f32 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as f32
    }
}

impl FromScalar<usize> for f64 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        a as f64
    }
}

impl FromScalar<usize> for f16 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<usize> for Complex32 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<usize> for Complex64 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<usize> for bf16 {
    #[inline(always)]
    fn __from(a: usize) -> Self {
        bf16::from_f64(a as f64)
    }
}

impl FromScalar<i8> for bool {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a != 0
    }
}

impl FromScalar<i8> for u8 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as u8
    }
}

impl FromScalar<i8> for u16 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as u16
    }
}

impl FromScalar<i8> for u32 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as u32
    }
}

impl FromScalar<i8> for u64 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as u64
    }
}

impl FromScalar<i8> for usize {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as usize
    }
}

impl FromScalar<i8> for i8 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a
    }
}

impl FromScalar<i8> for i16 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as i16
    }
}

impl FromScalar<i8> for i32 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as i32
    }
}

impl FromScalar<i8> for i64 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as i64
    }
}

impl FromScalar<i8> for isize {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as isize
    }
}

impl FromScalar<i8> for f32 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as f32
    }
}

impl FromScalar<i8> for f64 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        a as f64
    }
}

impl FromScalar<i8> for f16 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<i8> for Complex32 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<i8> for Complex64 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<i8> for bf16 {
    #[inline(always)]
    fn __from(a: i8) -> Self {
        bf16::from_f32(a as f32)
    }
}

impl FromScalar<i16> for bool {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a != 0
    }
}

impl FromScalar<i16> for u8 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as u8
    }
}

impl FromScalar<i16> for u16 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as u16
    }
}

impl FromScalar<i16> for u32 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as u32
    }
}

impl FromScalar<i16> for u64 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as u64
    }
}

impl FromScalar<i16> for usize {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as usize
    }
}

impl FromScalar<i16> for i8 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as i8
    }
}

impl FromScalar<i16> for i16 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a
    }
}

impl FromScalar<i16> for i32 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as i32
    }
}

impl FromScalar<i16> for i64 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as i64
    }
}

impl FromScalar<i16> for isize {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as isize
    }
}

impl FromScalar<i16> for f32 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as f32
    }
}

impl FromScalar<i16> for f64 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        a as f64
    }
}

impl FromScalar<i16> for f16 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<i16> for Complex32 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<i16> for Complex64 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<i16> for bf16 {
    #[inline(always)]
    fn __from(a: i16) -> Self {
        bf16::from_f32(a as f32)
    }
}

impl FromScalar<i32> for bool {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a != 0
    }
}

impl FromScalar<i32> for u8 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as u8
    }
}

impl FromScalar<i32> for u16 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as u16
    }
}

impl FromScalar<i32> for u32 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as u32
    }
}

impl FromScalar<i32> for u64 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as u64
    }
}

impl FromScalar<i32> for usize {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as usize
    }
}

impl FromScalar<i32> for i8 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as i8
    }
}

impl FromScalar<i32> for i16 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as i16
    }
}

impl FromScalar<i32> for i32 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a
    }
}

impl FromScalar<i32> for i64 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as i64
    }
}

impl FromScalar<i32> for isize {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as isize
    }
}

impl FromScalar<i32> for f32 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as f32
    }
}

impl FromScalar<i32> for f64 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        a as f64
    }
}

impl FromScalar<i32> for f16 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<i32> for Complex32 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<i32> for Complex64 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<i32> for bf16 {
    #[inline(always)]
    fn __from(a: i32) -> Self {
        bf16::from_f32(a as f32)
    }
}

impl FromScalar<i64> for bool {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a != 0
    }
}

impl FromScalar<i64> for u8 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as u8
    }
}

impl FromScalar<i64> for u16 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as u16
    }
}

impl FromScalar<i64> for u32 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as u32
    }
}

impl FromScalar<i64> for u64 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as u64
    }
}

impl FromScalar<i64> for usize {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as usize
    }
}

impl FromScalar<i64> for i8 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as i8
    }
}

impl FromScalar<i64> for i16 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as i16
    }
}

impl FromScalar<i64> for i32 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as i32
    }
}

impl FromScalar<i64> for i64 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a
    }
}

impl FromScalar<i64> for isize {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as isize
    }
}

impl FromScalar<i64> for f32 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as f32
    }
}

impl FromScalar<i64> for f64 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        a as f64
    }
}

impl FromScalar<i64> for f16 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<i64> for Complex32 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<i64> for Complex64 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<i64> for bf16 {
    #[inline(always)]
    fn __from(a: i64) -> Self {
        bf16::from_f64(a as f64)
    }
}

impl FromScalar<isize> for bool {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a != 0
    }
}

impl FromScalar<isize> for u8 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as u8
    }
}

impl FromScalar<isize> for u16 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as u16
    }
}

impl FromScalar<isize> for u32 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as u32
    }
}

impl FromScalar<isize> for u64 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as u64
    }
}

impl FromScalar<isize> for usize {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as usize
    }
}

impl FromScalar<isize> for i8 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as i8
    }
}

impl FromScalar<isize> for i16 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as i16
    }
}

impl FromScalar<isize> for i32 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as i32
    }
}

impl FromScalar<isize> for i64 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as i64
    }
}

impl FromScalar<isize> for isize {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a
    }
}

impl FromScalar<isize> for f32 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as f32
    }
}

impl FromScalar<isize> for f64 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        a as f64
    }
}

impl FromScalar<isize> for f16 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<isize> for Complex32 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<isize> for Complex64 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<isize> for bf16 {
    #[inline(always)]
    fn __from(a: isize) -> Self {
        bf16::from_f64(a as f64)
    }
}

impl FromScalar<f32> for bool {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a != 0.0
    }
}

impl FromScalar<f32> for u8 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as u8
    }
}

impl FromScalar<f32> for u16 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as u16
    }
}

impl FromScalar<f32> for u32 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as u32
    }
}

impl FromScalar<f32> for u64 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as u64
    }
}

impl FromScalar<f32> for usize {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as usize
    }
}

impl FromScalar<f32> for i8 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as i8
    }
}

impl FromScalar<f32> for i16 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as i16
    }
}

impl FromScalar<f32> for i32 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as i32
    }
}

impl FromScalar<f32> for i64 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as i64
    }
}

impl FromScalar<f32> for isize {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as isize
    }
}

impl FromScalar<f32> for f32 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a
    }
}

impl FromScalar<f32> for f64 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        a as f64
    }
}

impl FromScalar<f32> for f16 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        f16::from_f32(a)
    }
}

impl FromScalar<f32> for Complex32 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        Complex32::new(a, 0.0)
    }
}

impl FromScalar<f32> for Complex64 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        Complex64::new(a as f64, 0.0)
    }
}

impl FromScalar<f32> for bf16 {
    #[inline(always)]
    fn __from(a: f32) -> Self {
        bf16::from_f32(a)
    }
}

impl FromScalar<f64> for bool {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a != 0.0
    }
}

impl FromScalar<f64> for u8 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as u8
    }
}

impl FromScalar<f64> for u16 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as u16
    }
}

impl FromScalar<f64> for u32 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as u32
    }
}

impl FromScalar<f64> for u64 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as u64
    }
}

impl FromScalar<f64> for usize {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as usize
    }
}

impl FromScalar<f64> for i8 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as i8
    }
}

impl FromScalar<f64> for i16 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as i16
    }
}

impl FromScalar<f64> for i32 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as i32
    }
}

impl FromScalar<f64> for i64 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as i64
    }
}

impl FromScalar<f64> for isize {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as isize
    }
}

impl FromScalar<f64> for f32 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a as f32
    }
}

impl FromScalar<f64> for f64 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        a
    }
}

impl FromScalar<f64> for f16 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        f16::from_f32(a as f32)
    }
}

impl FromScalar<f64> for Complex32 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        Complex32::new(a as f32, 0.0)
    }
}

impl FromScalar<f64> for Complex64 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        Complex64::new(a, 0.0)
    }
}

impl FromScalar<f64> for bf16 {
    #[inline(always)]
    fn __from(a: f64) -> Self {
        bf16::from_f64(a)
    }
}

impl FromScalar<f16> for bool {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a != f16::ZERO
    }
}

impl FromScalar<f16> for u8 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as u8
    }
}

impl FromScalar<f16> for u16 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as u16
    }
}

impl FromScalar<f16> for u32 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as u32
    }
}

impl FromScalar<f16> for u64 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as u64
    }
}

impl FromScalar<f16> for usize {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as usize
    }
}

impl FromScalar<f16> for i8 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as i8
    }
}

impl FromScalar<f16> for i16 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as i16
    }
}

impl FromScalar<f16> for i32 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as i32
    }
}

impl FromScalar<f16> for i64 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as i64
    }
}

impl FromScalar<f16> for isize {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as isize
    }
}

impl FromScalar<f16> for f32 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32()
    }
}

impl FromScalar<f16> for f64 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a.to_f32() as f64
    }
}

impl FromScalar<f16> for f16 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        a
    }
}

impl FromScalar<f16> for Complex32 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        Complex32::new(a.to_f32(), 0.0)
    }
}

impl FromScalar<f16> for Complex64 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        Complex64::new(a.to_f32() as f64, 0.0)
    }
}

impl FromScalar<f16> for bf16 {
    #[inline(always)]
    fn __from(a: f16) -> Self {
        bf16::from_f32(a.to_f32())
    }
}

impl FromScalar<Complex32> for bool {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a != Complex32::ZERO
    }
}

impl FromScalar<Complex32> for u8 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as u8
    }
}

impl FromScalar<Complex32> for u16 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as u16
    }
}

impl FromScalar<Complex32> for u32 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as u32
    }
}

impl FromScalar<Complex32> for u64 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as u64
    }
}

impl FromScalar<Complex32> for usize {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as usize
    }
}

impl FromScalar<Complex32> for i8 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as i8
    }
}

impl FromScalar<Complex32> for i16 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as i16
    }
}

impl FromScalar<Complex32> for i32 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as i32
    }
}

impl FromScalar<Complex32> for i64 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as i64
    }
}

impl FromScalar<Complex32> for isize {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as isize
    }
}

impl FromScalar<Complex32> for f32 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm()
    }
}

impl FromScalar<Complex32> for f64 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a.norm() as f64
    }
}

impl FromScalar<Complex32> for f16 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        f16::from_f32(a.norm())
    }
}

impl FromScalar<Complex32> for Complex32 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        a
    }
}

impl FromScalar<Complex32> for Complex64 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        Complex64::new(a.re as f64, a.im as f64)
    }
}

impl FromScalar<Complex64> for bool {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a != Complex64::ZERO
    }
}

impl FromScalar<Complex64> for u8 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as u8
    }
}

impl FromScalar<Complex64> for u16 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as u16
    }
}

impl FromScalar<Complex64> for u32 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as u32
    }
}

impl FromScalar<Complex64> for u64 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as u64
    }
}

impl FromScalar<Complex64> for usize {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as usize
    }
}

impl FromScalar<Complex64> for i8 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as i8
    }
}

impl FromScalar<Complex64> for i16 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as i16
    }
}

impl FromScalar<Complex64> for i32 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as i32
    }
}

impl FromScalar<Complex64> for i64 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as i64
    }
}

impl FromScalar<Complex64> for isize {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as isize
    }
}

impl FromScalar<Complex64> for f32 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm() as f32
    }
}

impl FromScalar<Complex64> for f64 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a.norm()
    }
}

impl FromScalar<Complex64> for f16 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        f16::from_f32(a.norm() as f32)
    }
}

impl FromScalar<Complex64> for Complex32 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        Complex32::new(a.re as f32, a.im as f32)
    }
}

impl FromScalar<Complex64> for Complex64 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        a
    }
}

impl FromScalar<bf16> for bool {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a != bf16::ZERO
    }
}

impl FromScalar<bf16> for u8 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as u8
    }
}

impl FromScalar<bf16> for u16 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as u16
    }
}

impl FromScalar<bf16> for u32 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as u32
    }
}

impl FromScalar<bf16> for u64 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as u64
    }
}

impl FromScalar<bf16> for usize {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as usize
    }
}

impl FromScalar<bf16> for i8 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as i8
    }
}

impl FromScalar<bf16> for i16 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as i16
    }
}

impl FromScalar<bf16> for i32 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as i32
    }
}

impl FromScalar<bf16> for i64 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as i64
    }
}

impl FromScalar<bf16> for isize {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as isize
    }
}

impl FromScalar<bf16> for f32 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32()
    }
}

impl FromScalar<bf16> for f64 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a.to_f32() as f64
    }
}

impl FromScalar<bf16> for f16 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        f16::from_f32(a.to_f32())
    }
}

impl FromScalar<bf16> for Complex32 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        Complex32::new(a.to_f32(), 0.0)
    }
}

impl FromScalar<bf16> for Complex64 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        Complex64::new(a.to_f32() as f64, 0.0)
    }
}

impl FromScalar<bf16> for bf16 {
    #[inline(always)]
    fn __from(a: bf16) -> Self {
        a
    }
}

impl FromScalar<Complex32> for bf16 {
    #[inline(always)]
    fn __from(a: Complex32) -> Self {
        bf16::from_f32(a.norm())
    }
}

impl FromScalar<Complex64> for bf16 {
    #[inline(always)]
    fn __from(a: Complex64) -> Self {
        bf16::from_f64(a.norm())
    }
}
