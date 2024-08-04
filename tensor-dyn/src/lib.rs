pub mod tensor_base;
pub mod ops {
    pub mod cpu {
        pub mod binary_normal;
        pub mod std_ops;
        pub mod uary;
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
    }
}
pub mod backend;
pub mod tensor;
pub mod random;
pub mod slice;
pub mod to_tensor;
pub use tensor_traits::*;

use std::cell::RefCell;
thread_local! {
    static THREAD_POOL: RefCell<threadpool::ThreadPool> = RefCell::new(
        threadpool::ThreadPool::new(std::thread::available_parallelism().unwrap().into())
    );
}