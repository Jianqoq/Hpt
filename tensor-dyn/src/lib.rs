pub mod tensor_base;
pub mod ops {
    pub mod cpu {
        pub mod binary_normal;
        pub mod std_ops;
        pub mod unary;
        pub mod binary;
        pub mod matmul;
        pub mod reduce;
        pub mod reduce_kernels;
        pub mod tensordot;
        pub mod fft;
        pub mod cmp;
        pub mod stack;
        pub mod softmax;
        pub mod unique;
        pub mod hamming_window;
        pub mod hann_window;
        pub mod hardmax;
        pub mod blackman_window;
        pub mod onehot;
        pub mod affine_grid;
        pub mod pad;
        pub mod topk;
        pub mod shrink;
        pub mod gather_elements;
        pub mod gather;
        pub mod conv;
        pub mod maxpool;
        pub mod avgpool;
        pub mod lp_pool;
        pub mod dropout;
        pub mod lp_norm;
    }
    pub mod wgpu {
        pub mod buffer_helper;
        pub mod binary;
        pub mod binary_normal;
        pub mod std_ops;
        pub mod unary;
    }
}

pub mod wgpu_kernels {}

pub mod tensors {
    pub mod wgpu;
}

pub mod backend;
pub mod tensor;
pub mod random;
pub mod slice;
pub mod to_tensor;
pub mod wgpu_exec;
use ctor::ctor;
pub use tensor_iterator::iterator_traits::*;
pub use tensor_traits::*;

use std::{ cell::RefCell, sync::atomic::AtomicUsize };
thread_local! {
    static THREAD_POOL: RefCell<threadpool::ThreadPool> = RefCell::new(
        threadpool::ThreadPool::new(num_cpus::get_physical())
    );
}

static DISPLAY_PRECISION: AtomicUsize = AtomicUsize::new(4);
static DISPLAY_LR_ELEMENTS: AtomicUsize = AtomicUsize::new(3);
pub fn set_global_display_precision(precision: usize) {
    DISPLAY_PRECISION.store(precision, std::sync::atomic::Ordering::Relaxed);
}
pub fn set_global_display_lr_elements(lr_elements: usize) {
    DISPLAY_LR_ELEMENTS.store(lr_elements, std::sync::atomic::Ordering::Relaxed);
}

pub fn set_num_threads(num_threads: usize) {
    THREAD_POOL.with(|x| {
        x.borrow_mut().set_num_threads(num_threads);
    });
    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
}
pub fn get_num_threads() -> usize {
    THREAD_POOL.with(|x| x.borrow().max_count())
}

#[ctor]
fn init() {
    rayon::ThreadPoolBuilder::new().num_threads(num_cpus::get_physical()).build_global().unwrap();
}