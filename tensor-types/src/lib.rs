#![feature(portable_simd)]
pub mod dtype;
pub extern crate half;
pub mod type_promote;
pub mod convertion;
pub mod into_scalar;
pub mod into_vec;
pub mod vectors {
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
    pub mod traits;
}
pub use vectors::*;