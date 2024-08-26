use num_complex::{Complex32, Complex64};
use wide::*;


pub trait IntoVec<T> {
    fn into_vec(self) -> T;
}

impl IntoVec<f32x8> for f32x8 {
    fn into_vec(self) -> f32x8 {
        self
    }
}

impl IntoVec<[bool; 32]> for [bool; 32] {
    fn into_vec(self) -> [bool; 32] {
        self
    }
}

impl IntoVec<i8x32> for i8x32 {
    fn into_vec(self) -> i8x32 {
        self
    }
}

impl IntoVec<[u8; 32]> for [u8; 32] {
    fn into_vec(self) -> [u8; 32] {
        self
    }
}

impl IntoVec<i16x16> for i16x16 {
    fn into_vec(self) -> i16x16 {
        self
    }
}

impl IntoVec<u16x8> for u16x8 {
    fn into_vec(self) -> u16x8 {
        self
    }
}

impl IntoVec<i32x8> for i32x8 {
    fn into_vec(self) -> i32x8 {
        self
    }
}

impl IntoVec<u32x8> for u32x8 {
    fn into_vec(self) -> u32x8 {
        self
    }
}

impl IntoVec<i64x4> for i64x4 {
    fn into_vec(self) -> i64x4 {
        self
    }
}

impl IntoVec<u64x2> for u64x2 {
    fn into_vec(self) -> u64x2 {
        self
    }
}

impl IntoVec<f64x4> for f64x4 {
    fn into_vec(self) -> f64x4 {
        self
    }
}

impl IntoVec<[isize; 8]> for [isize; 8] {
    fn into_vec(self) -> [isize; 8] {
        self
    }
}

impl IntoVec<[usize; 8]> for [usize; 8] {
    fn into_vec(self) -> [usize; 8] {
        self
    }
}

impl IntoVec<[Complex32; 4]> for [Complex32; 4] {
    fn into_vec(self) -> [Complex32; 4] {
        self
    }
}

impl IntoVec<[Complex64; 2]> for [Complex64; 2] {
    fn into_vec(self) -> [Complex64; 2] {
        self
    }
}

impl IntoVec<[half::f16; 32]> for [half::f16; 32] {
    fn into_vec(self) -> [half::f16; 32] {
        self
    }
}

impl IntoVec<[half::bf16; 32]> for [half::bf16; 32] {
    fn into_vec(self) -> [half::bf16; 32] {
        self
    }
}