pub(crate) mod ops;
pub(crate) mod tensor;
pub(crate) mod utils;
use std::{cell::RefCell, sync::atomic::AtomicUsize};

pub use hpt_types::dtype::DType;
pub use tensor::Tensor;
pub use utils::device::Device;

static DISPLAY_PRECISION: AtomicUsize = AtomicUsize::new(4);
static DISPLAY_LR_ELEMENTS: AtomicUsize = AtomicUsize::new(4);
static ALIGN: usize = 64;

thread_local! {
    static THREAD_POOL: RefCell<crate::utils::thread_pool::ComputeThreadPool> = RefCell::new(
        crate::utils::thread_pool::ComputeThreadPool::new(num_cpus::get_physical())
    );
}

pub fn current_num_threads() -> usize {
    THREAD_POOL.with(|x| x.borrow().num_threads())
}

#[ctor::ctor]
fn init() {
    THREAD_POOL.with(|x| {
        x.borrow_mut().resize(num_cpus::get_physical());
    });
}

pub mod onnx {
    pub use crate::utils::onnx::load_model::load_onnx;
    pub(crate) use crate::utils::onnx::proto::*;
}
