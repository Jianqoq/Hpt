pub(crate) mod ops;
pub(crate) mod tensor;
pub(crate) mod utils;
use std::sync::atomic::AtomicUsize;

pub use hpt_types::dtype::DType;
pub use tensor::Tensor;
pub use utils::device::Device;

static DISPLAY_PRECISION: AtomicUsize = AtomicUsize::new(4);
static DISPLAY_LR_ELEMENTS: AtomicUsize = AtomicUsize::new(4);
static ALIGN: usize = 128;

pub fn current_num_threads() -> usize {
    rayon::current_num_threads()
}

pub fn physical_cores() -> usize {
    num_cpus::get_physical()
}

pub mod onnx {
    pub use crate::utils::onnx::load_model::load_onnx;
    pub(crate) use crate::utils::onnx::proto::*;
}
