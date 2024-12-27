//! This crate implment type utilities for tensor operations

#![cfg_attr(feature = "stdsimd", feature(portable_simd))]
#![deny(missing_docs)]

/// A module defines a set of data types and utilities
pub mod dtype;
pub extern crate half;
/// A module implement type conversion
pub mod convertion;
/// A module implement type conversion
pub mod into_scalar;
/// A module implement simd vector conversion
pub mod into_vec;
/// A module defines a set of traits for tensor operations, and implement computation functions for scalar and vector types
pub mod type_promote;
/// A module defines a set of traits for scalar operations
pub(crate) mod scalars {
    pub(crate) mod _bf16;
    pub(crate) mod _bool;
    pub(crate) mod _cplx32;
    pub(crate) mod _cplx64;
    pub(crate) mod _f16;
    pub(crate) mod _f32;
    pub(crate) mod _f64;
    pub(crate) mod _i16;
    pub(crate) mod _i32;
    pub(crate) mod _i64;
    pub(crate) mod _i8;
    pub(crate) mod _isize;
    pub(crate) mod _u16;
    pub(crate) mod _u32;
    pub(crate) mod _u64;
    pub(crate) mod _u8;
    pub(crate) mod _usize;
}

/// A module defines a set of traits for type promotion
pub mod promotion {
    pub(crate) mod _bf16;
    pub(crate) mod _bool;
    pub(crate) mod _cplx32;
    pub(crate) mod _cplx64;
    pub(crate) mod _f16;
    pub(crate) mod _f32;
    pub(crate) mod _f64;
    pub(crate) mod _i16;
    pub(crate) mod _i32;
    pub(crate) mod _i64;
    pub(crate) mod _i8;
    pub(crate) mod _isize;
    pub(crate) mod _u16;
    pub(crate) mod _u32;
    pub(crate) mod _u64;
    pub(crate) mod _u8;
    pub(crate) mod _usize;
    pub(crate) mod utils;
}

/// A module defines a set of vector types
pub mod vectors {
    /// A module defines a set of vector types using stdsimd
    #[cfg(feature = "stdsimd")]
    pub mod std_simd {
        /// A module defines a set of 128-bit vector types
        #[cfg(any(
            all(not(target_feature = "avx2"), target_feature = "sse"),
            target_arch = "arm",
            target_arch = "aarch64",
            target_feature = "neon"
        ))]
        pub mod _128bit {
            /// A module defines a set of 128-bit vector types for bf16
            pub mod bf16x8;
            /// A module defines a set of 128-bit vector types for bool
            pub mod boolx16;
            /// A module defines a set of 128-bit vector types for cplx32
            pub mod cplx32x2;
            /// A module defines a set of 128-bit vector types for cplx64
            pub mod cplx64x1;
            /// A module defines a set of 128-bit vector types for f16
            pub mod f16x8;
            /// A module defines a set of 128-bit vector types for f32
            pub mod f32x4;
            /// A module defines a set of 128-bit vector types for f64
            pub mod f64x2;
            /// A module defines a set of 128-bit vector types for i16
            pub mod i16x8;
            /// A module defines a set of 128-bit vector types for i32
            pub mod i32x4;
            /// A module defines a set of 128-bit vector types for i64
            pub mod i64x2;
            /// A module defines a set of 128-bit vector types for i8
            pub mod i8x16;
            /// A module defines a set of 128-bit vector types for isize
            pub mod isizex2;
            /// A module defines a set of 128-bit vector types for u16
            pub mod u16x8;
            /// A module defines a set of 128-bit vector types for u32
            pub mod u32x4;
            /// A module defines a set of 128-bit vector types for u64
            pub mod u64x2;
            /// A module defines a set of 128-bit vector types for u8
            pub mod u8x16;
            /// A module defines a set of 128-bit vector types for usize
            pub mod usizex2;
        }
        /// A module defines a set of 256-bit vector types
        #[cfg(target_feature = "avx2")]
        pub mod _256bit {
            /// A module defines a set of 256-bit vector types for bf16
            pub mod bf16x16;
            /// A module defines a set of 256-bit vector types for bool
            pub mod boolx32;
            /// A module defines a set of 256-bit vector types for cplx32
            pub mod cplx32x4;
            /// A module defines a set of 256-bit vector types for cplx64
            pub mod cplx64x2;
            /// A module defines a set of 256-bit vector types for f16
            pub mod f16x16;
            /// A module defines a set of 256-bit vector types for f32
            pub mod f32x8;
            /// A module defines a set of 256-bit vector types for f64
            pub mod f64x4;
            /// A module defines a set of 256-bit vector types for i16
            pub mod i16x16;
            /// A module defines a set of 256-bit vector types for i32
            pub mod i32x8;
            /// A module defines a set of 256-bit vector types for i64
            pub mod i64x4;
            /// A module defines a set of 256-bit vector types for i8
            pub mod i8x32;
            /// A module defines a set of 256-bit vector types for isize
            pub mod isizex4;
            /// A module defines a set of 256-bit vector types for u16
            pub mod u16x16;
            /// A module defines a set of 256-bit vector types for u32
            pub mod u32x8;
            /// A module defines a set of 256-bit vector types for u64
            pub mod u64x4;
            /// A module defines a set of 256-bit vector types for u8
            pub mod u8x32;
            /// A module defines a set of 256-bit vector types for usize
            pub mod usizex4;
        }
        /// A module defines a set of 512-bit vector types
        #[cfg(target_feature = "avx512f")]
        pub mod _512bit {
            /// A module defines a set of 512-bit vector types for bf16
            pub mod bf16x32;
            /// A module defines a set of 512-bit vector types for bool
            pub mod boolx64;
            /// A module defines a set of 512-bit vector types for cplx32
            pub mod cplx32x8;
            /// A module defines a set of 512-bit vector types for cplx64
            pub mod cplx64x4;
            /// A module defines a set of 512-bit vector types for f16
            pub mod f16x32;
            /// A module defines a set of 512-bit vector types for f32
            pub mod f32x16;
            /// A module defines a set of 512-bit vector types for f64
            pub mod f64x8;
            /// A module defines a set of 512-bit vector types for i16
            pub mod i16x32;
            /// A module defines a set of 512-bit vector types for i32
            pub mod i32x16;
            /// A module defines a set of 512-bit vector types for i64
            pub mod i64x8;
            /// A module defines a set of 512-bit vector types for i8
            pub mod i8x64;
            /// A module defines a set of 512-bit vector types for isize
            pub mod isizex8;
            /// A module defines a set of 512-bit vector types for u16
            pub mod u16x32;
            /// A module defines a set of 512-bit vector types for u32
            pub mod u32x16;
            /// A module defines a set of 512-bit vector types for u64
            pub mod u64x8;
            /// A module defines a set of 512-bit vector types for u8
            pub mod u8x64;
            /// A module defines a set of 512-bit vector types for usize
            pub mod usizex8;
        }
    }
    /// A module defines a set of vector types using stdsimd
    #[cfg(feature = "archsimd")]
    pub mod arch_simd {
        /// A module defines a set of 128-bit vector types
        #[cfg(any(
            all(not(target_feature = "avx2"), target_feature = "sse"),
            target_arch = "arm",
            target_arch = "aarch64",
            target_feature = "neon"
        ))]
        pub mod _128bit {
            /// A module defines a set of 128-bit vector types for bf16
            pub mod bf16x8;
            /// A module defines a set of 128-bit vector types for bool
            pub mod boolx16;
            /// A module defines a set of 128-bit vector types for cplx32
            pub mod cplx32x2;
            /// A module defines a set of 128-bit vector types for cplx64
            pub mod cplx64x1;
            /// A module defines a set of 128-bit vector types for f16
            pub mod f16x8;
            /// A module defines a set of 128-bit vector types for f32
            pub mod f32x4;
            /// A module defines a set of 128-bit vector types for f64
            pub mod f64x2;
            /// A module defines a set of 128-bit vector types for i16
            pub mod i16x8;
            /// A module defines a set of 128-bit vector types for i32
            pub mod i32x4;
            /// A module defines a set of 128-bit vector types for i64
            pub mod i64x2;
            /// A module defines a set of 128-bit vector types for i8
            pub mod i8x16;
            /// A module defines a set of 128-bit vector types for isize
            pub mod isizex2;
            /// A module defines a set of 128-bit vector types for u16
            pub mod u16x8;
            /// A module defines a set of 128-bit vector types for u32
            pub mod u32x4;
            /// A module defines a set of 128-bit vector types for u64
            pub mod u64x2;
            /// A module defines a set of 128-bit vector types for u8
            pub mod u8x16;
            /// A module defines a set of 128-bit vector types for usize
            pub mod usizex2;
        }
        /// A module defines a set of 256-bit vector types
        #[cfg(target_feature = "avx2")]
        pub mod _256bit {
            /// A module defines a set of 256-bit vector types for bf16
            pub mod bf16x16;
            /// A module defines a set of 256-bit vector types for bool
            pub mod boolx32;
            /// A module defines a set of 256-bit vector types for cplx32
            pub mod cplx32x4;
            /// A module defines a set of 256-bit vector types for cplx64
            pub mod cplx64x2;
            /// A module defines a set of 256-bit vector types for f16
            pub mod f16x16;
            /// A module defines a set of 256-bit vector types for f32
            pub mod f32x8;
            /// A module defines a set of 256-bit vector types for f64
            pub mod f64x4;
            /// A module defines a set of 256-bit vector types for i16
            pub mod i16x16;
            /// A module defines a set of 256-bit vector types for i32
            pub mod i32x8;
            /// A module defines a set of 256-bit vector types for i64
            pub mod i64x4;
            /// A module defines a set of 256-bit vector types for i8
            pub mod i8x32;
            /// A module defines a set of 256-bit vector types for isize
            pub mod isizex4;
            /// A module defines a set of 256-bit vector types for u16
            pub mod u16x16;
            /// A module defines a set of 256-bit vector types for u32
            pub mod u32x8;
            /// A module defines a set of 256-bit vector types for u64
            pub mod u64x4;
            /// A module defines a set of 256-bit vector types for u8
            pub mod u8x32;
            /// A module defines a set of 256-bit vector types for usize
            pub mod usizex4;
        }
        /// A module defines a set of 512-bit vector types
        #[cfg(target_feature = "avx512f")]
        pub mod _512bit {
            /// A module defines a set of 512-bit vector types for bf16
            pub mod bf16x32;
            /// A module defines a set of 512-bit vector types for bool
            pub mod boolx64;
            /// A module defines a set of 512-bit vector types for cplx32
            pub mod cplx32x8;
            /// A module defines a set of 512-bit vector types for cplx64
            pub mod cplx64x4;
            /// A module defines a set of 512-bit vector types for f16
            pub mod f16x32;
            /// A module defines a set of 512-bit vector types for f32
            pub mod f32x16;
            /// A module defines a set of 512-bit vector types for f64
            pub mod f64x8;
            /// A module defines a set of 512-bit vector types for i16
            pub mod i16x32;
            /// A module defines a set of 512-bit vector types for i32
            pub mod i32x16;
            /// A module defines a set of 512-bit vector types for i64
            pub mod i64x8;
            /// A module defines a set of 512-bit vector types for i8
            pub mod i8x64;
            /// A module defines a set of 512-bit vector types for isize
            pub mod isizex8;
            /// A module defines a set of 512-bit vector types for u16
            pub mod u16x32;
            /// A module defines a set of 512-bit vector types for u32
            pub mod u32x16;
            /// A module defines a set of 512-bit vector types for u64
            pub mod u64x8;
            /// A module defines a set of 512-bit vector types for u8
            pub mod u8x64;
            /// A module defines a set of 512-bit vector types for usize
            pub mod usizex8;
        }

        /// A module defines a set of vector types for sleef
        pub mod sleef {
            /// A module defines a set of vector types for table
            pub mod table;
            /// A module defines a set of vector types for helper
            pub mod arch {
                /// A module defines a set of vector types for helper
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                pub mod helper_aarch64;
                /// A module defines a set of vector types for helper
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                pub mod helper_avx2;
                /// A module defines a set of vector types for helper
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "sse",
                    not(target_feature = "avx2")
                ))]
                pub mod helper_sse;
            }
            /// A module defines a set of vector types for common
            pub mod common {
                /// A module defines a set of vector types for common
                pub mod commonfuncs;
                /// A module defines a set of vector types for common
                pub mod dd;
                /// A module defines a set of vector types for common
                pub mod df;
                /// A module defines a macro for polynomial approximation
                pub mod estrin;
                /// A module defines a set of vector types for common
                pub mod misc;
            }
            /// A module defines a set of vector types for libm
            pub mod libm {
                /// a module defins a set of double precision floating point functions
                pub mod sleefsimddp;
                /// a module defins a set of single precision floating point functions
                pub mod sleefsimdsp;
            }
        }
    }
    /// A module defines a set of traits for vector
    pub mod traits;
    /// A module defines a set of utils for vector
    pub mod utils;

    #[cfg(target_feature = "avx2")]
    pub(crate) mod vector_promote {
        #[cfg(target_pointer_width = "64")]
        pub(crate) use crate::vectors::arch_simd::_256bit::isizex4::isize_promote;
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_256bit::isizex8::isize_promote;
        #[cfg(target_pointer_width = "64")]
        pub(crate) use crate::vectors::arch_simd::_256bit::usizex4::usize_promote;
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_256bit::usizex8::usize_promote;
        pub(crate) use crate::vectors::arch_simd::_256bit::{
            bf16x16::bf16_promote, boolx32::bool_promote, cplx32x4::Complex32_promote,
            cplx64x2::Complex64_promote, f16x16::f16_promote, f32x8::f32_promote,
            f64x4::f64_promote, i16x16::i16_promote, i32x8::i32_promote, i64x4::i64_promote,
            i8x32::i8_promote, u16x16::u16_promote, u32x8::u32_promote, u64x4::u64_promote,
            u8x32::u8_promote,
        };
    }
    #[cfg(any(
        all(not(target_feature = "avx2"), target_feature = "sse"),
        target_arch = "arm",
        target_arch = "aarch64",
        target_feature = "neon"
    ))]
    pub(crate) mod vector_promote {
        #[cfg(target_pointer_width = "64")]
        pub(crate) use crate::vectors::arch_simd::_128bit::isizex2::isize_promote;
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_128bit::isizex4::isize_promote;
        #[cfg(target_pointer_width = "64")]
        pub(crate) use crate::vectors::arch_simd::_128bit::usizex2::usize_promote;
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_128bit::usizex4::usize_promote;
        pub(crate) use crate::vectors::arch_simd::_128bit::{
            bf16x8::bf16_promote, boolx16::bool_promote, cplx32x2::Complex32_promote,
            cplx64x1::Complex64_promote, f16x8::f16_promote, f32x4::f32_promote,
            f64x2::f64_promote, i16x8::i16_promote, i32x4::i32_promote, i64x2::i64_promote,
            i8x16::i8_promote, u16x8::u16_promote, u32x4::u32_promote, u64x2::u64_promote,
            u8x16::u8_promote,
        };
    }
    #[cfg(target_feature = "avx512f")]
    pub(crate) mod vector_promote {
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_512bit::isizex16::isize_promote;
        #[cfg(target_pointer_width = "64")]
        pub(crate) use crate::vectors::arch_simd::_512bit::isizex8::isize_promote;
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_512bit::usizex16::usize_promote;
        #[cfg(target_pointer_width = "64")]
        pub(crate) use crate::vectors::arch_simd::_512bit::usizex8::usize_promote;
        pub(crate) use crate::vectors::arch_simd::_512bit::{
            bf16x32::bf16_promote, boolx64::bool_promote, cplx32x8::Complex32_promote,
            cplx64x4::Complex64_promote, f16x32::f16_promote, f32x16::f32_promote,
            f64x8::f64_promote, i16x32::i16_promote, i32x16::i32_promote, i64x8::i64_promote,
            i8x64::i8_promote, u16x32::u16_promote, u32x16::u32_promote, u64x8::u64_promote,
            u8x64::u8_promote,
        };
    }
}

#[cfg(feature = "cuda")]
/// A module defines a set of types for cuda
pub mod cuda_types {
    /// A module defines convertion for cuda types
    pub mod convertion;
    /// A module defines a scalar type for cuda
    pub mod scalar;

    pub(crate) mod _bool;
    pub(crate) mod _cplx32;
    pub(crate) mod _cplx64;
    pub(crate) mod _f16;
    pub(crate) mod _bf16;
    pub(crate) mod _f32;
    pub(crate) mod _f64;
    pub(crate) mod _i16;
    pub(crate) mod _i32;
    pub(crate) mod _i64;
    pub(crate) mod _i8;
    pub(crate) mod _isize;
    pub(crate) mod _u16;
    pub(crate) mod _u32;
    pub(crate) mod _u64;
    pub(crate) mod _u8;
    pub(crate) mod _usize;
}

pub use vectors::*;
#[cfg(feature = "archsimd")]
mod simd {
    pub use crate::vectors::arch_simd::*;
}
#[cfg(feature = "stdsimd")]
mod simd {
    pub use crate::vectors::std_simd::*;
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(feature = "stdsimd")
))]
pub(crate) mod sleef_types {
    use std::arch::x86_64::*;
    pub(crate) type VDouble = __m256d;
    pub(crate) type VMask = __m256i;
    pub(crate) type Vopmask = __m256i;
    pub(crate) type VFloat = __m256;
    pub(crate) type VInt = __m128i;
    pub(crate) type VInt2 = __m256i;
    pub(crate) type VInt64 = __m256i;
    pub(crate) type VUInt64 = __m256i;
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse",
    not(target_feature = "avx2"),
    not(feature = "stdsimd")
))]
pub(crate) mod sleef_types {
    use std::arch::x86_64::*;
    pub(crate) type VDouble = __m128d;
    pub(crate) type VMask = __m128i;
    pub(crate) type Vopmask = __m128i;
    pub(crate) type VFloat = __m128;
    pub(crate) type VInt = __m128i;
    pub(crate) type VInt2 = __m128i;
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod sleef_types {
    use std::arch::aarch64::*;
    pub(crate) type VDouble = float64x2_t;
    pub(crate) type VMask = uint32x4_t;
    pub(crate) type Vopmask = uint32x4_t;
    pub(crate) type VFloat = float32x4_t;
    pub(crate) type VInt = int32x2_t;
    pub(crate) type VInt2 = int32x4_t;
}
