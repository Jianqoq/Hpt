
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 8 f32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct f32x8(#[cfg(target_arch = "x86_64")] pub(crate) __m256);