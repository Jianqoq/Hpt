
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 32 u8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct u8x64(#[cfg(target_arch = "x86_64")] pub(crate) __m512i);