
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 2 u64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u64x4(#[cfg(target_arch = "x86_64")] pub(crate) __m256i);
