//! This crate is dynamic graph based tensor libraryz
#![cfg_attr(feature = "stdsimd", feature(portable_simd))]
#![deny(missing_docs)]

/// a module contains all the Tensor operations. include the CPU and GPU operations
pub mod ops {
    /// a module contains all the CPU operations
    pub mod cpu {
        /// a module defines affine_grid operation
        pub mod affine_grid;
        /// a module defines all normal binary operation
        pub mod binary;
        /// a module defines binary normal iterations,
        /// the `binary` module uses this module's function to perform computation
        pub mod binary_normal;
        /// a module defines concat operation
        pub mod concat;
        /// a module defines conv2d operation
        pub mod conv2d {
            /// a module defines avgpool2d operation
            pub mod avg_pool;
            /// a module defines conv2d operation
            pub mod conv2d;
            /// a module defines conv2d_group operation
            pub mod conv2d_group;
            /// a module defines dwconv2d operation
            pub mod dwconv2d;
            /// a module defines maxpool2d operation
            pub mod max_pool;
        }
        /// a module defines reduce kernels
        pub mod argreduce_kernels;
        /// a module defines dropout operation
        pub mod dropout;
        /// a module defines gather operation
        pub mod gather;
        /// a module defines gather_elements operation
        pub mod gather_elements;
        /// a module defines hardmax operation
        pub mod hardmax;
        /// a module defines matmul operation
        pub mod matmul;
        /// a module defines max_roi_pool operation
        pub mod max_roi_pool;
        /// a module defines onehot operation
        pub mod onehot;
        /// a module defines pad operation
        pub mod pad;
        /// a module defines internal reduce functions
        pub mod reduce;
        /// a module defines reduce template
        pub mod reduce_template;
        /// a module contains all the reduce computation utils
        pub mod reduce_utils;
        /// a module defines shrink operation
        pub mod shrink;
        /// a module defines softmax operations
        pub mod softmaxs {
            /// a module defines log_softmax
            pub mod log_softmax;
            /// a module defines softmax
            pub mod softmax;
            /// a module defines softmax utils
            pub mod softmax_utils;
        }
        /// a module defines all the std::ops operations
        pub mod std_ops;
        /// a module defines tensordot operation
        pub mod tensordot;
        /// a module defines topk operation
        pub mod topk;
        /// a module defines all the unary operations
        pub mod unary;
        /// a module defines all the kernels
        pub mod kernels {
            /// a module defines the avgpool2d kernels
            pub mod avgpool_kernels;
            /// a module defines the conv2d kernels
            pub mod conv;
            /// a module defines the dwconv2d kernels
            pub mod conv_group;
            /// a module defines the dwconv2d kernels
            pub mod dwconv;
            /// a module defines the logsoftmax kernels
            pub mod logsoftmax;
            /// a module defines the lp_pool2d kernels
            pub mod lp_pool_kernels;
            /// a module defines the reduce kernels
            pub mod reduce;
            /// a module defines the softmax kernels
            pub mod softmax;
        }
        /// a module that contains all the functions expose for the external user (we may have diff tensor (differentiable tensor) in the future)
        pub mod tensor_external {
            /// a module that contains all the arg reduce functions
            pub mod arg_reduce;
            /// a module that contains all the tensor compare functions
            pub mod cmp;
            /// a module that contains all the common reduce functions
            pub mod common_reduce;
            /// a module that contains all fft operations
            pub mod fft;
            /// a module that contains all the unary operations that has floating type output
            pub mod float_out_unary;
            /// a module that contains matrix multiplication operations
            pub mod matmul;
            /// a module that contains all normal methods to create a tensor
            pub mod normal_creation;
            /// a module that contains all the unary operations that has self type output
            pub mod normal_out_unary;
            /// a module that contains all the random number generate functions
            pub mod random;
            /// a module that contains all the shape manipulation functions
            pub mod shape_manipulate;
            /// a module that contains all the slice functions
            pub mod slice;
            /// a module that contains all the windows creation functions
            pub mod windows;
        }
        /// a module that contains all the functions only for the internal user (we may have diff tensor (differentiable tensor) in the future)
        pub mod tensor_internal {
            /// a module that contains all the arg reduce functions
            pub mod arg_reduce;
            /// a module that contains all the tensor compare functions
            pub mod cmp;
            /// a module that contains all the common reduce functions
            pub mod common_reduce;
            /// a module that contains all fft operations
            pub mod fft;
            /// a module that contains all the unary operations that has floating type output
            pub mod float_out_unary;
            /// a module that contains matrix multiplication operations
            pub mod matmul;
            /// a module that contains all normal methods to create a tensor
            pub mod normal_creation;
            /// a module that contains all the unary operations that has self type output
            pub mod normal_out_unary;
            /// a module that contains all the random number generate functions
            pub mod random;
            /// a module that contains all the shape manipulation functions
            pub mod shape_manipulate;
            /// a module that contains all the slice functions
            pub mod slice;
            /// a module that contains all the windows creation functions
            pub mod windows;
        }

        /// a module contains cpu L1, L2, L3 cache helper
        pub(crate) mod cache_utils {
            /// a module contains cache utils
            pub(crate) mod cache;
        }
        /// a module contains cpu tensor impls
        pub(crate) mod tensor_impls;
    }

    /// a module contains cuda tensor impls
    pub(crate) mod cuda {
        /// a module contains cuda tensor impls
        pub(crate) mod tensor_impls;
        /// a module contains cuda tensor internal impls
        pub(crate) mod tensor_internal {
            /// a module contains cuda tensor normal creation impls
            pub(crate) mod normal_creation;
            /// a module contains cuda tensor shape manipulation impls
            pub(crate) mod shape_manipulate;
            /// a module contains cuda tensor common reduce impls
            pub(crate) mod common_reduce;
        }
        pub mod tensor_external {
            /// a module contains cuda tensor normal creation impls
            pub(crate) mod normal_creation;
            /// a module contains cuda tensor matmul impls
            pub(crate) mod matmul;
            /// a module contains cuda tensor shape manipulation impls
            pub(crate) mod shape_manipulate;
            /// a module contains cuda tensor common reduce impls
            pub(crate) mod common_reduce;
        }
        /// a module contains cuda utils
        pub(crate) mod cuda_utils;
        /// a module contains cuda kernel constants
        pub(crate) mod kernel_constants;
        /// a module contains cuda binary normal impls
        pub(crate) mod binary_normal;
        /// a module contains cuda std ops impls
        pub(crate) mod std_ops;
        /// a module contains cuda matmul impls
        pub(crate) mod matmul;
        /// a module contains cuda slice impls
        pub(crate) mod cuda_slice;
        /// a module contains cuda concat impls
        pub(crate) mod concat;
        /// a module contains cuda reduce impls
        pub(crate) mod reduce;
        /// a module contains cuda reduce template impls    
        pub(crate) mod reduce_template;
        /// a module contains cuda reduce utils impls
        pub(crate) mod reduce_utils;
    }
}

pub mod backend;
/// a module that wrap the _Tensor struct
pub mod tensor;
/// a module that defines the _Tensor struct
pub mod tensor_base;
/// a module that contains the implementation of the `Into` trait for the `_Tensor` struct.
///
/// # Note
/// for this library's developer, not necessary need to know how they works
pub mod to_tensor;
use ctor::ctor;
pub use tensor_iterator::iterator_traits::*;
pub use tensor_iterator::TensorIterator;

pub use crate::backend::*;
pub use rayon::prelude::*;
pub use tensor::Tensor;
pub use tensor_codegen::compile;
pub use tensor_codegen::fuse_proc_macro;
pub use tensor_macros::match_selection;
pub use tensor_traits::*;
pub use tensor_types::dtype::TypeCommon;
pub use tensor_types::traits::VecTrait;
pub use tensor_types::type_promote::*;
pub use tensor_types::vectors::*;
pub use tensor_types::*;

use std::{cell::RefCell, sync::atomic::AtomicUsize};
thread_local! {
    static THREAD_POOL: RefCell<threadpool::ThreadPool> = RefCell::new(
        threadpool::ThreadPool::new(num_cpus::get_physical())
    );
}

/// Set the Tensor display precision
pub fn set_global_display_precision(precision: usize) {
    DISPLAY_PRECISION.store(precision, std::sync::atomic::Ordering::Relaxed);
}

/// Set the left and right elements to display for each dimension
pub fn set_global_display_lr_elements(lr_elements: usize) {
    DISPLAY_LR_ELEMENTS.store(lr_elements, std::sync::atomic::Ordering::Relaxed);
}

/// Set the global number of threads
///
/// # Note
/// Rayon only allows the number of threads to be set once, so the rayon thread pool won't have any effect if it's called more than once.
pub fn set_num_threads(num_threads: usize) {
    THREAD_POOL.with(|x| {
        x.borrow_mut().set_num_threads(num_threads);
    });
    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .stack_size(4 * 1024 * 1024)
        .build_global()
    {
        Ok(_) => {}
        Err(_) => {}
    }
}

/// Get the global number of threads
pub fn get_num_threads() -> usize {
    THREAD_POOL.with(|x| x.borrow().max_count())
}

static DISPLAY_PRECISION: AtomicUsize = AtomicUsize::new(4);
static DISPLAY_LR_ELEMENTS: AtomicUsize = AtomicUsize::new(3);

#[cfg(feature = "cuda")]
pub(crate) mod cuda_compiled {
    use std::{collections::HashMap, sync::{Arc, Mutex}};

    use once_cell::sync::Lazy;

    use crate::ops::cuda::cuda_utils::RegisterInfo;
    pub(crate) static CUDA_COMPILED: Lazy<
        Mutex<HashMap<usize, HashMap<String, Arc<HashMap<String, RegisterInfo>>>>>,
    > = Lazy::new(|| Mutex::new(HashMap::new()));
}

#[ctor]
fn init() {
    THREAD_POOL.with(|x| {
        x.borrow_mut().set_num_threads(num_cpus::get_physical());
    });
}

static ALIGN: usize = 64;

#[cfg(target_feature = "avx2")]
pub(crate) const REGNUM: usize = 16;
#[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
pub(crate) const REGNUM: usize = 8;
#[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
pub(crate) const REGNUM: usize = 32;

#[cfg(feature = "archsimd")]
use tensor_types::arch_simd as simd;
#[cfg(feature = "stdsimd")]
use tensor_types::std_simd as simd;

#[cfg(target_feature = "avx2")]
type BoolVector = simd::_256bit::boolx32::boolx32;
#[cfg(any(target_feature = "avx512f"))]
type BoolVector = simd::_512bit::boolx64::boolx64;
#[cfg(any(
    all(not(target_feature = "avx2"), target_feature = "sse"),
    target_arch = "arm",
    target_arch = "aarch64",
    target_feature = "neon"
))]
type BoolVector = simd::_128bit::boolx16::boolx16;

#[cfg(target_feature = "avx2")]
const SIMD_WIDTH: usize = 256;
#[cfg(any(target_feature = "avx512f"))]
const SIMD_WIDTH: usize = 512;
#[cfg(any(
    all(not(target_feature = "avx2"), target_feature = "sse"),
    target_arch = "arm",
    target_arch = "aarch64",
    target_feature = "neon"
))]
const SIMD_WIDTH: usize = 128;
