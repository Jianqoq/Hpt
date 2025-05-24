
#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 4 f32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct f32x4(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128,
    #[cfg(target_arch = "aarch64")] pub(crate) float32x4_t,
);
