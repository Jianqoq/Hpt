
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::ops::Index;

/// a vector of 2 u64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u64x2(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")] pub(crate) uint64x2_t,
);

impl Index<usize> for u64x2 {
    type Output = u64;
    fn index(&self, index: usize) -> &Self::Output {
        let ptr = self as *const _ as *const u64;
        unsafe { &*ptr.add(index) }
    }
}