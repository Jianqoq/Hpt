pub mod tensor;
pub mod ops {
    pub mod binary_funcs_normal;
    pub mod std_ops;
    pub mod uary;
    pub mod binary;
    pub mod matmul;
}

pub mod to_tensor;

use std::cell::RefCell;
thread_local! {
    static THREAD_POOL: RefCell<threadpool::ThreadPool> = RefCell::new(
        threadpool::ThreadPool::new(std::thread::available_parallelism().unwrap().into())
    );
}