#![feature(portable_simd)]
pub mod tensor_base;
pub mod ops {
    pub mod cpu {
        pub mod affine_grid;
        pub mod avg_pool2d;
        pub mod binary;
        pub mod binary_normal;
        pub mod blackman_window;
        pub mod cmp;
        pub mod common_reduce;
        pub mod concat;
        pub mod conv2d;
        pub mod conv_config;
        pub mod dropout;
        pub mod fft;
        pub mod gather;
        pub mod gather_elements;
        pub mod hardmax;
        pub mod lp_norm;
        pub mod lp_pool2d;
        pub mod matmul;
        pub mod max_roi_pool;
        pub mod maxpool2d;
        pub mod onehot;
        pub mod pad;
        pub mod reduce;
        pub mod reduce_kernels;
        pub mod reduce_template;
        pub mod reduce_utils;
        pub mod shrink;
        pub mod softmax;
        pub mod std_ops;
        pub mod tensordot;
        pub mod topk;
        pub mod unary;
        pub mod unique;
        pub mod windows;
        pub mod kernels {
            pub mod avgpool_kernels;
            pub mod conv_kernels;
            pub mod lp_pool_kernels;
            pub mod maxpool_kernels;
            pub mod reduce_kernels;
        }
    }
}

pub mod wgpu_kernels {}

pub mod backend;
pub mod random;
pub mod slice;
pub mod tensor;
pub mod to_tensor;
pub mod wgpu_exec;
use ctor::ctor;
pub use tensor_iterator::iterator_traits::*;
pub use tensor_macros::match_selection;
pub use tensor_traits::*;
pub use tensor_types::vectors::*;
pub use tensor_types::*;

use std::{cell::RefCell, sync::atomic::AtomicUsize};
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
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .stack_size(4 * 1024 * 1024)
        .build_global()
        .unwrap();
}
pub fn get_num_threads() -> usize {
    THREAD_POOL.with(|x| x.borrow().max_count())
}

#[ctor]
fn init() {
    THREAD_POOL.with(|x| {
        x.borrow_mut().set_num_threads(num_cpus::get());
    });
}

static ALIGN: usize = 64;
#[cfg(target_feature = "avx2")]
pub(crate) const CONV_REGNUM: usize = 7;
#[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
pub(crate) const CONV_REGNUM: usize = 3;
#[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
pub(crate) const CONV_REGNUM: usize = 15;

#[cfg(target_feature = "avx2")]
pub(crate) const REGNUM: usize = 16;
#[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
pub(crate) const REGNUM: usize = 8;
#[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
pub(crate) const REGNUM: usize = 32;

#[cfg(target_feature = "avx2")]
type BoolVector = tensor_types::_256bit::boolx32::boolx32;
#[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
type BoolVector = tensor_types::_512bit::boolx64::boolx64;
#[cfg(any(
    all(not(target_feature = "avx2"), target_feature = "sse"),
    target_arch = "arm",
    target_arch = "aarch64",
    target_feature = "neon"
))]
type BoolVector = tensor_types::_128bit::boolx16::boolx16;
