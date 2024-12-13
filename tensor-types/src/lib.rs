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
}
pub use vectors::*;

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
