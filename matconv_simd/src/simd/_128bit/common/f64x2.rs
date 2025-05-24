
#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 2 f64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct f64x2(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128d,
    #[cfg(target_arch = "aarch64")] pub(crate) float64x2_t,
);
