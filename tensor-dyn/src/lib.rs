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
        pub mod gather;
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
pub use tensor_iterator::iterator_traits::*;
pub use tensor_traits::*;

use std::{ cell::RefCell, sync::atomic::AtomicUsize };
thread_local! {
    static THREAD_POOL: RefCell<threadpool::ThreadPool> = RefCell::new(
        threadpool::ThreadPool::new(std::thread::available_parallelism().unwrap().into())
    );
}

static mut DISPLAY_PRECISION: AtomicUsize = AtomicUsize::new(4);
static mut DISPLAY_LR_ELEMENTS: AtomicUsize = AtomicUsize::new(3);
pub fn set_global_display_precision(precision: usize) {
    unsafe {
        DISPLAY_PRECISION.store(precision, std::sync::atomic::Ordering::Relaxed);
    }
}
pub fn set_global_display_lr_elements(lr_elements: usize) {
    unsafe {
        DISPLAY_LR_ELEMENTS.store(lr_elements, std::sync::atomic::Ordering::Relaxed);
    }
}
