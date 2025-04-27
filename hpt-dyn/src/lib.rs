pub(crate) mod ops;
pub(crate) mod utils;
pub(crate) mod tensor;
use std::sync::atomic::AtomicUsize;

pub use tensor::Tensor;
pub use utils::device::Device;
pub use hpt_types::dtype::DType;

static DISPLAY_PRECISION: AtomicUsize = AtomicUsize::new(4);
static DISPLAY_LR_ELEMENTS: AtomicUsize = AtomicUsize::new(3);

pub fn current_num_threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(target_feature = "avx2")]
pub(crate) const REGNUM: usize = 16;
#[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
pub(crate) const REGNUM: usize = 8;
#[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
pub(crate) const REGNUM: usize = 32;