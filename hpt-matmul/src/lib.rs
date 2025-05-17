#![cfg_attr(any(target_feature = "avx512f"), feature(stdarch_x86_avx512))]

pub(crate) mod template;
pub(crate) mod type_kernels {
    #[cfg(feature = "bf16")]
    pub(crate) mod bf16_kernels;
    #[cfg(feature = "bool")]
    pub(crate) mod bool_kernels;
    pub(crate) mod common;
    #[cfg(feature = "cplx32")]
    pub(crate) mod cplx32_kernels;
    #[cfg(feature = "cplx64")]
    pub(crate) mod cplx64_kernels;
    #[cfg(feature = "f16")]
    pub(crate) mod f16_kernels;
    #[cfg(feature = "f32")]
    pub(crate) mod f32_kernels;
    #[cfg(feature = "f64")]
    pub(crate) mod f64_kernels;
    #[cfg(feature = "i16")]
    pub(crate) mod i16_kernels;
    #[cfg(feature = "i32")]
    pub(crate) mod i32_kernels;
    #[cfg(feature = "i64")]
    pub(crate) mod i64_kernels;
    #[cfg(feature = "i8")]
    pub(crate) mod i8_kernels;
    #[cfg(feature = "u16")]
    pub(crate) mod u16_kernels;
    #[cfg(feature = "u32")]
    pub(crate) mod u32_kernels;
    #[cfg(feature = "u64")]
    pub(crate) mod u64_kernels;
    #[cfg(feature = "u8")]
    pub(crate) mod u8_kernels;
}

pub(crate) mod matmul;
pub(crate) mod microkernel_trait;
pub(crate) mod microkernels;
pub(crate) mod utils;

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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
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

pub(crate) const fn vec_size<T>() -> usize {
    REG_BITS / 8 / std::mem::size_of::<T>()
}

#[derive(Debug, Clone, Copy)]
pub struct Pointer<T> {
    ptr: *mut T,
    #[cfg(feature = "bound_check")]
    len: i64,
}

impl<T> Pointer<T> {
    #[allow(unused)]
    #[inline(always)]
    pub(crate) fn new(ptr: *mut T, len: i64) -> Self {
        Self {
            ptr,
            #[cfg(feature = "bound_check")]
            len,
        }
    }
    #[inline(always)]
    pub(crate) fn cast<U>(self) -> Pointer<U> {
        #[cfg(feature = "bound_check")]
        {
            let new_len =
                ((self.len as usize) * std::mem::size_of::<T>()) / std::mem::size_of::<U>();
            return Pointer::new(self.ptr as *mut U, new_len as i64);
        }
        #[cfg(not(feature = "bound_check"))]
        return Pointer::new(self.ptr as *mut U, 0);
    }
    #[inline(always)]
    pub(crate) fn offset(&self, offset: i64) -> Pointer<T> {
        unsafe {
            #[cfg(feature = "bound_check")]
            {
                if offset < 0 || offset >= (self.len as i64) {
                    panic!("index out of bounds. index: {}, len: {}", offset, self.len);
                }
                Pointer::new(self.ptr.offset(offset as isize), self.len - offset)
            }
            #[cfg(not(feature = "bound_check"))]
            {
                Pointer::new(self.ptr.offset(offset as isize), 0)
            }
        }
    }
    #[inline(always)]
    pub(crate) fn read_unaligned(&self) -> T {
        unsafe {
            std::ptr::read_unaligned(self.ptr)
        }
    }
}

impl<T> Index<i64> for Pointer<T> {
    type Output = T;

    fn index(&self, index: i64) -> &Self::Output {
        #[cfg(feature = "bound_check")]
        {
            if index < 0 || index >= (self.len as i64) {
                panic!("index out of bounds. index: {}, len: {}", index, self.len);
            }
        }
        unsafe { &*self.ptr.offset(index as isize) }
    }
}

impl<T> IndexMut<i64> for Pointer<T> {
    fn index_mut(&mut self, index: i64) -> &mut Self::Output {
        #[cfg(feature = "bound_check")]
        {
            if index < 0 || index >= (self.len as i64) {
                panic!("index out of bounds. index: {}, len: {}", index, self.len);
            }
        }
        unsafe { &mut *self.ptr.offset(index as isize) }
    }
}

impl<T> AddAssign<i64> for Pointer<T> {
    fn add_assign(&mut self, rhs: i64) {
        #[cfg(feature = "bound_check")]
        {
            self.len -= rhs as i64;
            assert!(self.len >= 0);
        }
        unsafe {
            self.ptr = self.ptr.offset(rhs as isize);
        }
    }
}

impl<T> std::ops::Add<i64> for Pointer<T> {
    type Output = Pointer<T>;

    fn add(self, rhs: i64) -> Self::Output {
        #[cfg(feature = "bound_check")]
        unsafe {
            Pointer::new(self.ptr.offset(rhs as isize), self.len - rhs)
        }
        #[cfg(not(feature = "bound_check"))]
        {
            Pointer::<T> {
                ptr: unsafe {
                    self.ptr.offset(rhs as isize)
                },
                #[cfg(feature = "bound_check")]
                len: self.len,
            }
        }
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

pub(crate) const REG_BITS: usize = std::mem::size_of::<F32Vec>() * 8;

use std::ops::{ AddAssign, Deref, DerefMut, Index, IndexMut };

#[cfg(
    any(
        all(not(target_feature = "avx2"), target_feature = "sse"),
        target_arch = "arm",
        target_arch = "aarch64",
        target_feature = "neon"
    )
)]
pub use crate::simd::_128bit::{
    Bf16Vec,
    BoolVec,
    Cplx32Vec,
    Cplx64Vec,
    F16Vec,
    F32Vec,
    F64Vec,
    I8Vec,
    I16Vec,
    I32Vec,
    I64Vec,
    U8Vec,
    U16Vec,
    U32Vec,
    U64Vec,
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
pub use crate::simd::_256bit::{
    Bf16Vec,
    BoolVec,
    Cplx32Vec,
    Cplx64Vec,
    F16Vec,
    F32Vec,
    F64Vec,
    I8Vec,
    I16Vec,
    I32Vec,
    I64Vec,
    U8Vec,
    U16Vec,
    U32Vec,
    U64Vec,
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use crate::simd::_512bit::{
    Bf16Vec,
    BoolVec,
    Cplx32Vec,
    Cplx64Vec,
    F16Vec,
    F32Vec,
    F64Vec,
    I8Vec,
    I16Vec,
    I32Vec,
    I64Vec,
    U8Vec,
    U16Vec,
    U32Vec,
    U64Vec,
};

pub use crate::microkernel_trait::MatmulMicroKernel;

pub trait Zero {
    const ZERO: Self;
}

#[allow(unused)]
pub(crate) trait Add {
    fn add(self, other: Self) -> Self;
}

pub use matmul::{matmul, prepack_rhs, addmm};

pub use utils::{ kernel_params, NewPrePackedRhs };

static ALIGN: usize = 128;
