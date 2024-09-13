#![feature(portable_simd)]
pub mod dtype;
pub extern crate half;
pub mod type_promote;
pub mod convertion;
pub mod into_scalar;
pub mod into_vec;
pub mod vectors {
    // #[cfg(all(any(target_feature = "sse", target_feature = "neon"), not(target_feature = "avx2")))]
    pub mod _128bit {
        pub mod f32x4;
        pub mod u32x4;
        pub mod i8x16;
        pub mod bf16x8;
        pub mod u16x8;
        pub mod boolx16;
        pub mod i16x8;
        pub mod i32x4;
        pub mod f64x2;
        pub mod i64x2;
        pub mod u64x2;
        pub mod u8x16;
        pub mod usizex2;
        pub mod isizex2;
        pub mod f16x8;
        pub mod cplx32x2;
        pub mod cplx64x1;
    }
    #[cfg(target_feature = "avx2")]
    pub mod _256bit {
        pub mod f32x8;
        pub mod f64x4;
        pub mod f16x16;
        pub mod i32x8;
        pub mod i64x4;
        pub mod i16x16;
        pub mod i8x32;
        pub mod boolx32;
        pub mod u8x32;
        pub mod u16x16;
        pub mod u32x8;
        pub mod u64x4;
        pub mod isizex4;
        pub mod usizex4;
        pub mod bf16x16;
        pub mod cplx32x4;
        pub mod cplx64x2;
    }
    #[cfg(target_feature = "avx512f")]
    pub mod _512bit {
        pub mod f32x16;
        pub mod f64x8;
        pub mod i32x16;
        pub mod i64x8;
        pub mod i16x32;
        pub mod i8x64;
        pub mod boolx64;
        pub mod u8x64;
        pub mod u16x32;
        pub mod u32x16;
        pub mod u64x8;
        pub mod isizex8;
        pub mod usizex8;
        pub mod f16x32;
        pub mod bf16x32;
        pub mod cplx32x8;
        pub mod cplx64x4;
    }
    pub mod traits;
}
pub use vectors::*;