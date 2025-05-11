pub(crate) mod template;
pub(crate) mod type_kernels {
    pub(crate) mod f32_kernels;
    pub(crate) mod f16_kernels;
    pub(crate) mod bf16_kernels;
    pub(crate) mod f64_kernels;
    pub(crate) mod i8_kernels;
    pub(crate) mod u8_kernels;
    pub(crate) mod i16_kernels;
    pub(crate) mod u16_kernels;
    pub(crate) mod i32_kernels;
    pub(crate) mod u32_kernels;
    pub(crate) mod i64_kernels;
    pub(crate) mod u64_kernels;
    pub(crate) mod cplx32_kernels;
    pub(crate) mod cplx64_kernels;
    pub(crate) mod bool_kernels;
    pub(crate) mod common;
}

pub(crate) mod utils;
pub(crate) mod microkernels;
pub(crate) mod microkernel_trait;
pub(crate) mod matmul;

pub(crate) mod simd {
    #[cfg(
        any(
            all(not(target_feature = "avx2"), target_feature = "sse"),
            target_arch = "arm",
            target_arch = "aarch64",
            target_feature = "neon"
        )
    )]
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
            /// the number of registers available
            pub const REGNUM: usize = 8;
        }

        pub(crate) type F32Vec = crate::simd::_128bit::common::f32x4::f32x4;
        pub(crate) type F64Vec = crate::simd::_128bit::common::f64x2::f64x2;
        pub(crate) type I16Vec = crate::simd::_128bit::common::i16x8::i16x8;
        pub(crate) type I32Vec = crate::simd::_128bit::common::i32x4::i32x4;
        pub(crate) type I64Vec = crate::simd::_128bit::common::i64x2::i64x2;
        pub(crate) type I8Vec = crate::simd::_128bit::common::i8x16::i8x16;
        pub(crate) type U16Vec = crate::simd::_128bit::common::u16x8::u16x8;
        pub(crate) type U32Vec = crate::simd::_128bit::common::u32x4::u32x4;
        pub(crate) type U64Vec = crate::simd::_128bit::common::u64x2::u64x2;
        pub(crate) type U8Vec = crate::simd::_128bit::common::u8x16::u8x16;
        pub(crate) type F16Vec = crate::simd::_128bit::common::f16x8::f16x8;
        pub(crate) type Bf16Vec = crate::simd::_128bit::common::bf16x8::bf16x8;
        pub(crate) type BoolVec = crate::simd::_128bit::common::boolx16::boolx16;
        pub(crate) type Cplx32Vec = crate::simd::_128bit::common::cplx32x2::cplx32x2;
        pub(crate) type Cplx64Vec = crate::simd::_128bit::common::cplx64x1::cplx64x1;
    }
    /// A module defines a set of 256-bit vector types
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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
            /// the number of registers available
            pub const REGNUM: usize = 16;
        }
    }

    pub(crate) type F32Vec = crate::simd::_256bit::common::f32x8::f32x8;
    pub(crate) type F64Vec = crate::simd::_256bit::common::f64x4::f64x4;
    pub(crate) type I16Vec = crate::simd::_256bit::common::i16x16::i16x16;
    pub(crate) type I32Vec = crate::simd::_256bit::common::i32x8::i32x8;
    pub(crate) type I64Vec = crate::simd::_256bit::common::i64x4::i64x4;
    pub(crate) type I8Vec = crate::simd::_256bit::common::i8x32::i8x32;
    pub(crate) type U16Vec = crate::simd::_256bit::common::u16x16::u16x16;
    pub(crate) type U32Vec = crate::simd::_256bit::common::u32x8::u32x8;
    pub(crate) type U64Vec = crate::simd::_256bit::common::u64x4::u64x4;
    pub(crate) type U8Vec = crate::simd::_256bit::common::u8x32::u8x32;
    pub(crate) type F16Vec = crate::simd::_256bit::common::f16x16::f16x16;
    pub(crate) type Bf16Vec = crate::simd::_256bit::common::bf16x16::bf16x16;
    pub(crate) type BoolVec = crate::simd::_256bit::common::boolx32::boolx32;
    pub(crate) type Cplx32Vec = crate::simd::_256bit::common::cplx32x4::cplx32x4;
    pub(crate) type Cplx64Vec = crate::simd::_256bit::common::cplx64x2::cplx64x2;
}

pub(crate) const fn vec_size<T>() -> usize {
    REG_BITS / 8 / std::mem::size_of::<T>()
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Pointer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Pointer<T> {
    pub(crate) fn new(ptr: *mut T, len: usize) -> Self {
        Self { ptr, len }
    }
    pub(crate) fn cast<U>(self) -> Pointer<U> {
        Pointer::<U> { ptr: self.ptr as *mut U, len: self.len }
    }
}

impl<T> Index<i64> for Pointer<T> {
    type Output = T;

    fn index(&self, index: i64) -> &Self::Output {
        unsafe { &*self.ptr.offset(index as isize) }
    }
}

impl<T> IndexMut<i64> for Pointer<T> {
    fn index_mut(&mut self, index: i64) -> &mut Self::Output {
        unsafe { &mut *self.ptr.offset(index as isize) }
    }
}

impl<T> AddAssign<i64> for Pointer<T> {
    fn add_assign(&mut self, rhs: i64) {
        unsafe {
            self.ptr = self.ptr.offset(rhs as isize);
        }
    }
}

impl<T> std::ops::Add<i64> for Pointer<T> {
    type Output = Pointer<T>;

    fn add(self, rhs: i64) -> Self::Output {
        Pointer::<T> { ptr: unsafe { self.ptr.offset(rhs as isize) }, len: self.len }
    }
}

impl<T> Deref for Pointer<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl<T> DerefMut for Pointer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.ptr }
    }
}

unsafe impl<T> Send for Pointer<T> {}
unsafe impl<T> Sync for Pointer<T> {}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) const REG_BITS: usize = 256;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) const REG_BITS: usize = 512;

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2"), target_feature = "sse"))]
pub(crate) const REG_BITS: usize = 128;

#[cfg(all(target_arch = "arm", target_feature = "neon"))]
pub(crate) const REG_BITS: usize = 128;

use std::ops::{ AddAssign, Deref, DerefMut, Index, IndexMut };

pub(crate) use crate::simd::{
    F16Vec,
    Bf16Vec,
    BoolVec,
    Cplx32Vec,
    Cplx64Vec,
    F32Vec,
    F64Vec,
    I16Vec,
    I32Vec,
    I64Vec,
    I8Vec,
    U16Vec,
    U32Vec,
    U64Vec,
    U8Vec,
};

pub(crate) trait Zero {
    const ZERO: Self;
}

pub(crate) trait Add {
    fn add(self, other: Self) -> Self;
}

pub use matmul::matmul;