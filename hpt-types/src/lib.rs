//! This crate implment type utilities for tensor operations
#![deny(missing_docs)]

/// A module implement type conversion
pub mod convertion;
/// A module defines a set of data types and utilities
pub mod dtype;
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
    pub(crate) mod _f16;
    pub(crate) mod _f32;
    pub(crate) mod _f64;
    pub(crate) mod impls;
}

/// A module defines a set of traits for type promotion
pub mod promotion {
    #[cfg(feature = "normal_promote")]
    pub(crate) mod normal_promote {
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
    pub(crate) mod utils;
}

/// A module defines a set of vector types
pub mod vectors {
    /// A module defines a set of vector types using stdsimd
    pub mod arch_simd {
        /// A module defines a set of 128-bit vector types
        #[cfg(any(
            all(not(target_feature = "avx2"), target_feature = "sse"),
            target_arch = "arm",
            target_arch = "aarch64",
            target_feature = "neon"
        ))]
        pub mod _128bit {
            pub(crate) mod common {
                pub(crate) mod bf16x8;
                pub(crate) mod boolx16;
                pub(crate) mod cplx32x2;
                pub(crate) mod cplx64x1;
                pub(crate) mod f16x8;
                pub(crate) mod f32x4;
                pub(crate) mod f64x2;
                pub(crate) mod i16x8;
                pub(crate) mod i32x4;
                pub(crate) mod i64x2;
                pub(crate) mod i8x16;
                pub(crate) mod isizex2;
                pub(crate) mod u16x8;
                pub(crate) mod u32x4;
                pub(crate) mod u64x2;
                pub(crate) mod u8x16;
                pub(crate) mod usizex2;
            }

            #[cfg(target_feature = "neon")]
            #[cfg(target_arch = "aarch64")]
            pub(crate) mod neon {
                pub(crate) mod bf16x8;
                pub(crate) mod boolx16;
                pub(crate) mod f16x8;
                pub(crate) mod f32x4;
                pub(crate) mod f64x2;
                pub(crate) mod i16x8;
                pub(crate) mod i32x4;
                pub(crate) mod i64x2;
                pub(crate) mod i8x16;
                pub(crate) mod u16x8;
                pub(crate) mod u32x4;
                pub(crate) mod u64x2;
                pub(crate) mod u8x16;
            }

            #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
            pub(crate) mod sse {
                pub(crate) mod bf16x8;
                pub(crate) mod boolx16;
                pub(crate) mod f16x8;
                pub(crate) mod f32x4;
                pub(crate) mod f64x2;
                pub(crate) mod i16x8;
                pub(crate) mod i32x4;
                pub(crate) mod i64x2;
                pub(crate) mod i8x16;
                pub(crate) mod u16x8;
                pub(crate) mod u32x4;
                pub(crate) mod u64x2;
                pub(crate) mod u8x16;
            }

            pub use crate::arch_simd::_128bit::common::{
                bf16x8::bf16x8, boolx16::boolx16, cplx32x2::cplx32x2, cplx64x1::cplx64x1,
                f16x8::f16x8, f32x4::f32x4, f64x2::f64x2, i16x8::i16x8, i32x4::i32x4, i64x2::i64x2,
                i8x16::i8x16, isizex2::isizex2, u16x8::u16x8, u32x4::u32x4, u64x2::u64x2,
                u8x16::u8x16, usizex2::usizex2,
            };
        }
        /// A module defines a set of 256-bit vector types
        #[cfg(target_feature = "avx2")]
        pub mod _256bit {
            pub(crate) mod common {
                pub(crate) mod bf16x16;
                pub(crate) mod boolx32;
                pub(crate) mod cplx32x4;
                pub(crate) mod cplx64x2;
                pub(crate) mod f16x16;
                pub(crate) mod f32x8;
                pub(crate) mod f64x4;
                pub(crate) mod i16x16;
                pub(crate) mod i32x8;
                pub(crate) mod i64x4;
                pub(crate) mod i8x32;
                pub(crate) mod isizex4;
                pub(crate) mod u16x16;
                pub(crate) mod u32x8;
                pub(crate) mod u64x4;
                pub(crate) mod u8x32;
                pub(crate) mod usizex4;
            }
            #[cfg(target_feature = "avx2")]
            pub(crate) mod avx2 {
                pub(crate) mod bf16x16;
                pub(crate) mod f16x16;
                pub(crate) mod f32x8;
                pub(crate) mod f64x4;
                pub(crate) mod i16x16;
                pub(crate) mod i32x8;
                pub(crate) mod i64x4;
                pub(crate) mod i8x32;
                pub(crate) mod u16x16;
                pub(crate) mod u32x8;
                pub(crate) mod u64x4;
                pub(crate) mod u8x32;
            }

            pub use crate::arch_simd::_256bit::common::{
                bf16x16::bf16x16, boolx32::boolx32, cplx32x4::cplx32x4, cplx64x2::cplx64x2,
                f16x16::f16x16, f32x8::f32x8, f64x4::f64x4, i16x16::i16x16, i32x8::i32x8,
                i64x4::i64x4, i8x32::i8x32, isizex4::isizex4, u16x16::u16x16, u32x8::u32x8,
                u64x4::u64x4, u8x32::u8x32, usizex4::usizex4,
            };
        }

        // This file contains code ported from SLEEF (https://github.com/shibatch/sleef)
        //
        // Original work Copyright (c) 2010-2022, Naoki Shibata and contributors
        // Modified work Copyright (c) 2024 hpt Contributors
        //
        // Boost Software License - Version 1.0 - August 17th, 2003
        //
        // Permission is hereby granted, free of charge, to any person or organization
        // obtaining a copy of the software and accompanying documentation covered by
        // this license (the "Software") to use, reproduce, display, distribute,
        // execute, and transmit the Software, and to prepare derivative works of the
        // Software, and to permit third-parties to whom the Software is furnished to
        // do so, all subject to the following:
        //
        // The copyright notices in the Software and this entire statement, including
        // the above license grant, this restriction and the following disclaimer,
        // must be included in all copies of the Software, in whole or in part, and
        // all derivative works of the Software, unless such copies or derivative
        // works are solely in the form of machine-executable object code generated by
        // a source language processor.
        //
        // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        // FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
        // SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
        // FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
        // ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
        // DEALINGS IN THE SOFTWARE.
        //
        // This Rust port is additionally licensed under Apache-2.0 OR MIT
        // See repository root for details
        /// A module defines a set of vector types for sleef
        #[allow(clippy::approx_constant)]
        #[allow(clippy::excessive_precision)]
        #[allow(clippy::unreadable_literal)]
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
        pub(crate) use crate::vectors::arch_simd::_256bit::common::isizex4::isize_promote;
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_256bit::common::isizex8::isize_promote;
        #[cfg(target_pointer_width = "64")]
        pub(crate) use crate::vectors::arch_simd::_256bit::common::usizex4::usize_promote;
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_256bit::common::usizex8::usize_promote;
        pub(crate) use crate::vectors::arch_simd::_256bit::common::{
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
        pub(crate) use crate::vectors::arch_simd::_128bit::common::isizex2::isize_promote;
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_128bit::common::isizex4::isize_promote;
        #[cfg(target_pointer_width = "64")]
        pub(crate) use crate::vectors::arch_simd::_128bit::common::usizex2::usize_promote;
        #[cfg(target_pointer_width = "32")]
        pub(crate) use crate::vectors::arch_simd::_128bit::common::usizex4::usize_promote;
        pub(crate) use crate::vectors::arch_simd::_128bit::common::{
            bf16x8::bf16_promote, boolx16::bool_promote, cplx32x2::Complex32_promote,
            cplx64x1::Complex64_promote, f16x8::f16_promote, f32x4::f32_promote,
            f64x2::f64_promote, i16x8::i16_promote, i32x4::i32_promote, i64x2::i64_promote,
            i8x16::i8_promote, u16x8::u16_promote, u32x4::u32_promote, u64x2::u64_promote,
            u8x16::u8_promote,
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

pub use vectors::*;
mod simd {
    pub use crate::vectors::arch_simd::*;
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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
    not(target_feature = "avx2")
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
