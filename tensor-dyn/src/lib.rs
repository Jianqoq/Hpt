
#![feature(portable_simd)]
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
        pub mod concat;
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
        pub mod maxpool;
        pub mod avgpool;
        pub mod lp_pool;
        pub mod dropout;
        pub mod lp_norm;
        pub mod img2col;
        pub mod common_reduce;
        pub mod convolutions {
            pub mod conv2d_test;
            pub mod conv2d;
            pub mod conv3d;
            pub mod conv_config;
        }
        pub mod kernels {
            pub mod conv2d_kernels;
            pub mod reduce_kernels;
        }
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
pub use tensor_macros::match_selection;
pub use tensor_types::vectors::*;

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
    rayon::ThreadPoolBuilder
        ::new()
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
#[cfg(target_feature = "avx512f")]
pub(crate) const CONV_REGNUM: usize = 15;