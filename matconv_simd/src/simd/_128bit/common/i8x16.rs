
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::ops::Index;

/// a vector of 16 i8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct i8x16(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")] pub(crate) int8x16_t,
);

impl Index<usize> for i8x16 {
    type Output = i8;
    fn index(&self, index: usize) -> &Self::Output {
        let ptr = self as *const _ as *const i8;
        unsafe { &*ptr.add(index) }
    }
}
