pub(crate) mod ops;
pub(crate) mod utils;
pub(crate) mod tensor;
use std::{cell::RefCell, sync::atomic::AtomicUsize};

pub use tensor::Tensor;
pub use utils::device::Device;
pub use hpt_types::dtype::DType;

static DISPLAY_PRECISION: AtomicUsize = AtomicUsize::new(4);
static DISPLAY_LR_ELEMENTS: AtomicUsize = AtomicUsize::new(3);
static ALIGN: usize = 64;

thread_local! {
    static THREAD_POOL: RefCell<crate::utils::thread_pool::ComputeThreadPool> = RefCell::new(
        crate::utils::thread_pool::ComputeThreadPool::new(num_cpus::get())
    );
}

pub fn current_num_threads() -> usize {
    rayon::current_num_threads()
}
