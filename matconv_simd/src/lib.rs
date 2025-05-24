#![cfg_attr(any(target_feature = "avx512f"), feature(stdarch_x86_avx512))]

pub(crate) mod simd {
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
            pub(crate) mod u16x8;
            pub(crate) mod u32x4;
            pub(crate) mod u64x2;
            pub(crate) mod u8x16;
        }

        #[cfg(target_feature = "neon")]
        #[cfg(target_arch = "aarch64")]
        pub(crate) mod neon {
            pub(crate) mod bf16x8;
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

        pub type F32Vec = crate::simd::_128bit::common::f32x4::f32x4;
        pub type F64Vec = crate::simd::_128bit::common::f64x2::f64x2;
        pub type I16Vec = crate::simd::_128bit::common::i16x8::i16x8;
        pub type I32Vec = crate::simd::_128bit::common::i32x4::i32x4;
        pub type I64Vec = crate::simd::_128bit::common::i64x2::i64x2;
        pub type I8Vec = crate::simd::_128bit::common::i8x16::i8x16;
        pub type U16Vec = crate::simd::_128bit::common::u16x8::u16x8;
        pub type U32Vec = crate::simd::_128bit::common::u32x4::u32x4;
        pub type U64Vec = crate::simd::_128bit::common::u64x2::u64x2;
        pub type U8Vec = crate::simd::_128bit::common::u8x16::u8x16;
        pub type F16Vec = crate::simd::_128bit::common::f16x8::f16x8;
        pub type Bf16Vec = crate::simd::_128bit::common::bf16x8::bf16x8;
        pub type BoolVec = crate::simd::_128bit::common::boolx16::boolx16;
        pub type Cplx32Vec = crate::simd::_128bit::common::cplx32x2::cplx32x2;
        pub type Cplx64Vec = crate::simd::_128bit::common::cplx64x1::cplx64x1;
    }
    /// A module defines a set of 256-bit vector types
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
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
            pub(crate) mod u16x16;
            pub(crate) mod u32x8;
            pub(crate) mod u64x4;
            pub(crate) mod u8x32;
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
        pub type F32Vec = crate::simd::_256bit::common::f32x8::f32x8;
        pub type F64Vec = crate::simd::_256bit::common::f64x4::f64x4;
        pub type I16Vec = crate::simd::_256bit::common::i16x16::i16x16;
        pub type I32Vec = crate::simd::_256bit::common::i32x8::i32x8;
        pub type I64Vec = crate::simd::_256bit::common::i64x4::i64x4;
        pub type I8Vec = crate::simd::_256bit::common::i8x32::i8x32;
        pub type U16Vec = crate::simd::_256bit::common::u16x16::u16x16;
        pub type U32Vec = crate::simd::_256bit::common::u32x8::u32x8;
        pub type U64Vec = crate::simd::_256bit::common::u64x4::u64x4;
        pub type U8Vec = crate::simd::_256bit::common::u8x32::u8x32;
        pub type F16Vec = crate::simd::_256bit::common::f16x16::f16x16;
        pub type Bf16Vec = crate::simd::_256bit::common::bf16x16::bf16x16;
        pub type BoolVec = crate::simd::_256bit::common::boolx32::boolx32;
        pub type Cplx32Vec = crate::simd::_256bit::common::cplx32x4::cplx32x4;
        pub type Cplx64Vec = crate::simd::_256bit::common::cplx64x2::cplx64x2;
    }

    /// A module defines a set of 256-bit vector types
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    pub mod _512bit {
        pub(crate) mod common {
            pub(crate) mod bf16x32;
            pub(crate) mod boolx64;
            pub(crate) mod cplx32x8;
            pub(crate) mod cplx64x4;
            pub(crate) mod f16x32;
            pub(crate) mod f32x16;
            pub(crate) mod f64x8;
            pub(crate) mod i16x32;
            pub(crate) mod i32x16;
            pub(crate) mod i64x8;
            pub(crate) mod i8x64;
            pub(crate) mod u16x32;
            pub(crate) mod u32x16;
            pub(crate) mod u64x8;
            pub(crate) mod u8x64;
            pub(crate) mod mask;
        }
        #[cfg(target_feature = "avx512f")]
        pub(crate) mod avx512 {
            pub(crate) mod bf16x32;
            pub(crate) mod f16x32;
            pub(crate) mod f32x16;
            pub(crate) mod f64x8;
            pub(crate) mod i16x32;
            pub(crate) mod i32x16;
            pub(crate) mod i64x8;
            pub(crate) mod i8x64;
            pub(crate) mod u16x32;
            pub(crate) mod u32x16;
            pub(crate) mod u64x8;
            pub(crate) mod u8x64;
        }
        pub type F32Vec = crate::simd::_512bit::common::f32x16::f32x16;
        pub type F64Vec = crate::simd::_512bit::common::f64x8::f64x8;
        pub type I16Vec = crate::simd::_512bit::common::i16x32::i16x32;
        pub type I32Vec = crate::simd::_512bit::common::i32x16::i32x16;
        pub type I64Vec = crate::simd::_512bit::common::i64x8::i64x8;
        pub type I8Vec = crate::simd::_512bit::common::i8x64::i8x64;
        pub type U16Vec = crate::simd::_512bit::common::u16x32::u16x32;
        pub type U32Vec = crate::simd::_512bit::common::u32x16::u32x16;
        pub type U64Vec = crate::simd::_512bit::common::u64x8::u64x8;
        pub type U8Vec = crate::simd::_512bit::common::u8x64::u8x64;
        pub type F16Vec = crate::simd::_512bit::common::f16x32::f16x32;
        pub type Bf16Vec = crate::simd::_512bit::common::bf16x32::bf16x32;
        pub type BoolVec = crate::simd::_512bit::common::boolx64::boolx64;
        pub type Cplx32Vec = crate::simd::_512bit::common::cplx32x8::cplx32x8;
        pub type Cplx64Vec = crate::simd::_512bit::common::cplx64x4::cplx64x4;
    }
}

#[cfg(any(
    all(not(target_feature = "avx2"), target_feature = "sse"),
    target_arch = "arm",
    target_arch = "aarch64",
    target_feature = "neon"
))]
pub use crate::simd::_128bit::{
    Bf16Vec, BoolVec, Cplx32Vec, Cplx64Vec, F16Vec, F32Vec, F64Vec, I8Vec, I16Vec, I32Vec, I64Vec,
    U8Vec, U16Vec, U32Vec, U64Vec,
};

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use crate::simd::_256bit::{
    Bf16Vec, BoolVec, Cplx32Vec, Cplx64Vec, F16Vec, F32Vec, F64Vec, I8Vec, I16Vec, I32Vec, I64Vec,
    U8Vec, U16Vec, U32Vec, U64Vec,
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use crate::simd::_512bit::{
    Bf16Vec, BoolVec, Cplx32Vec, Cplx64Vec, F16Vec, F32Vec, F64Vec, I8Vec, I16Vec, I32Vec, I64Vec,
    U8Vec, U16Vec, U32Vec, U64Vec,
};

pub(crate) const REG_BITS: usize = std::mem::size_of::<F32Vec>() * 8;

pub trait VecTrait<T> {
    const SIZE: usize = REG_BITS / 8 / std::mem::size_of::<T>();
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn splat(val: T) -> Self;
    #[allow(unused)]
    fn partial_load(ptr: *const T, num_elem: usize) -> Self;
    #[allow(unused)]
    fn partial_store(self, ptr: *mut T, num_elem: usize);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self;
}

pub trait Zero {
    const ZERO: Self;
}

impl Zero for half::f16 {
    const ZERO: Self = half::f16::from_f32_const(0.0);
}

impl Zero for half::bf16 {
    const ZERO: Self = half::bf16::from_f32_const(0.0);
}

impl Zero for f32 {
    const ZERO: Self = 0.0;
}

impl Zero for f64 {
    const ZERO: Self = 0.0;
}

impl Zero for i8 {
    const ZERO: Self = 0;
}

impl Zero for i16 {
    const ZERO: Self = 0;
}

impl Zero for i32 {
    const ZERO: Self = 0;
}

impl Zero for i64 {
    const ZERO: Self = 0;
}

#[allow(unused)]
pub(crate) trait Add {
    fn add(self, other: Self) -> Self;
}