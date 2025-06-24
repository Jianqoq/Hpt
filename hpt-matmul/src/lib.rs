

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

pub(crate) const fn vec_size<T>() -> usize {
    REG_BITS / 8 / std::mem::size_of::<T>()
}

#[derive(Debug, Clone, Copy)]
pub struct Pointer<T> {
    pub ptr: *mut T,
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
        unsafe { std::ptr::read_unaligned(self.ptr) }
    }
    #[inline(always)]
    pub(crate) fn write_unaligned(&self, val: T) {
        unsafe {
            std::ptr::write_unaligned(self.ptr, val);
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
                ptr: unsafe { self.ptr.offset(rhs as isize) },
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

use std::ops::{AddAssign, Deref, DerefMut, Index, IndexMut};

pub use crate::microkernel_trait::MatmulMicroKernel;

pub(crate) use matconv_simd::*;
pub use matmul::{addmm, matmul, prepack_rhs};

pub use utils::{PrePackedRhs, kernel_params};

static ALIGN: usize = 128;