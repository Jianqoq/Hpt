
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 4 i64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct i64x8(#[cfg(target_arch = "x86_64")] pub(crate) __m512i);
