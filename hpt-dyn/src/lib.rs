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

