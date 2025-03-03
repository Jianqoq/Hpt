//! This crate is dynamic graph based tensor library
#![deny(missing_docs)]

/// a module contains all the Tensor operations. include the CPU and GPU operations
pub mod ops {
    /// a module contains all the CPU operations
    pub mod cpu {
        /// a module defines affine_grid operation
        pub mod affine_grid;
        pub(crate) mod utils {
            pub(crate) mod reduce {
                pub(crate) mod reduce;
                pub(crate) mod reduce_template;
                pub(crate) mod reduce_utils;
            }
            pub(crate) mod diff {
                pub(crate) mod diff_utils;
            }
            pub(crate) mod binary {
                pub(crate) mod binary_normal;
            }
            pub(crate) mod unary {
                pub(crate) mod unary;
            }
        }
        /// a module defines all the std::ops operations
        pub mod std_ops;
        /// a module defines all the kernels
        pub mod kernels {
            /// a module defines reduce kernels
            pub mod argreduce_kernels;
            /// a module defines the batchnorm conv2d kernels
            pub mod batch_norm_conv;
            /// a module defines the conv2d kernels
            pub mod conv;
            /// a module defines the dwconv2d kernels
            pub mod conv_group;
            /// a module defines the conv transpose kernels
            pub mod conv_transpose;
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
            /// a module contains all the pooling operations
            pub mod pooling {
                /// a module contains all the common pooling operations
                pub mod common;
            }
            /// a module defines normalization operations
            pub mod normalization {
                /// a module defines layernorm
                pub mod layernorm;
                /// a module defines log_softmax
                pub mod log_softmax;
                /// a module defines softmax utils
                pub mod normalize_utils;
                /// a module defines softmax
                pub mod softmax;
            }
            /// a module defines conv2d operation
            pub mod conv2d {
                /// a module defines batchnorm_conv2d operation
                pub mod batchnorm_conv2d;
                /// a module defines conv2d operation
                pub mod conv2d;
                /// a module defines conv2d_group operation
                pub mod conv2d_group;
                /// a module defines conv2d_transpose operation
                pub mod conv2d_transpose;
                /// a module defines dwconv2d operation
                pub mod dwconv2d;
            }
        }
        /// a module that contains all the functions expose for the external user (we may have diff tensor (differentiable tensor) in the future)
        pub mod tensor_external {
            /// a module that contains all the advance operations
            pub mod advance;
            /// a module that contains all the arg reduce functions
            pub mod arg_reduce;
            /// a module defines all normal binary operation
            pub mod binary;
            /// a module that contains all the tensor compare functions
            pub mod cmp;
            /// a module that contains all the common reduce functions
            pub mod common_reduce;
            /// a module that contains all the conv functions
            pub mod conv;
            /// a module that contains all the cumulative operations
            pub mod cumulative;
            /// a module that contains all fft operations
            pub mod fft;
            /// a module that contains all the float out binary operations
            pub mod float_out_binary;
            /// a module that contains all the unary operations that has floating type output
            pub mod float_out_unary;
            /// a module that contains matrix multiplication operations
            pub mod matmul;
            /// a module that contains all normal methods to create a tensor
            pub mod normal_creation;
            /// a module that contains all the unary operations that has self type output
            pub mod normal_out_unary;
            /// a module that contains all the pooling functions
            pub mod pooling;
            /// a module that contains all the random number generate functions
            pub mod random;
            /// a module that contains all the shape manipulation functions
            pub mod shape_manipulate;
            /// a module that contains all the slice functions
            pub mod slice;
            /// a module that contains all the tensordot functions
            pub mod tensordot;
            /// a module that contains all the windows creation functions
            pub mod windows;
        }
        /// a module that contains all the functions only for the internal user (we may have diff tensor (differentiable tensor) in the future)
        pub mod tensor_internal {
            /// a module that contains all the advance operations
            pub mod advance;
            /// a module that contains all the arg reduce functions
            pub mod arg_reduce;
            /// a module that contains all the tensor compare functions
            pub mod cmp;
            /// a module that contains all the common reduce functions
            pub mod common_reduce;
            /// a module that contains all the conv functions
            pub mod conv;
            /// a module that contains all the cumulative operations
            pub mod cumulative;
            /// a module that contains all fft operations
            pub mod fft;
            /// a module that contains all the float out binary operations
            pub mod float_out_binary;
            /// a module that contains all the unary operations that has floating type output
            pub mod float_out_unary;
            /// a module that contains matrix multiplication operations
            pub mod matmul;
            /// a module that contains all normal methods to create a tensor
            pub mod normal_creation;
            /// a module that contains all the unary operations that has self type output
            pub mod normal_out_unary;
            /// a module that contains all the pooling functions
            pub mod pooling;
            /// a module that contains all the random number generate functions
            pub mod random;
            /// a module that contains all the shape manipulation functions
            pub mod shape_manipulate;
            /// a module that contains all the tensordot functions
            pub mod tensordot;
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

    #[cfg(feature = "cuda")]
    /// a module contains cuda tensor impls
    pub(crate) mod cuda {
        /// a module contains cuda tensor impls
        pub(crate) mod tensor_impls;
        /// a module contains cuda tensor internal impls
        pub(crate) mod tensor_internal {
            /// a module contains cuda tensor advanced impls
            pub(crate) mod advance;
            /// a module contains cuda tensor arg reduce impls
            pub(crate) mod arg_reduce;
            /// a module contains cuda tensor common reduce impls
            pub(crate) mod common_reduce;
            /// a module contains cuda tensor float out binary impls
            pub(crate) mod float_out_binary;
            /// a module contains cuda tensor float out unary impls
            pub(crate) mod float_out_unary;
            /// a module contains cuda matmul impls
            pub(crate) mod matmul;
            /// a module contains cuda tensor normal creation impls
            pub(crate) mod normal_creation;
            /// a module contains cuda tensor normal out unary impls
            pub(crate) mod normal_out_unary;
            /// a module contains cuda tensor random impls
            pub(crate) mod random;
            /// a module contains cuda tensor shape manipulation impls
            pub(crate) mod shape_manipulate;
            /// a module contains cuda tensor windows impls
            pub(crate) mod windows;
        }
        pub mod tensor_external {
            /// a module contains cuda tensor arg reduce impls
            pub(crate) mod arg_reduce;
            /// a module contains cuda tensor cmp impls
            pub(crate) mod cmp;
            /// a module contains cuda tensor common reduce impls
            pub(crate) mod common_reduce;
            /// a module contains cuda tensor float out binary impls
            pub(crate) mod float_out_binary;
            /// a module contains cuda tensor float out unary impls
            pub(crate) mod float_out_unary;
            /// a module contains cuda tensor matmul impls
            pub(crate) mod matmul;
            /// a module contains cuda tensor normal creation impls
            pub(crate) mod normal_creation;
            /// a module contains cuda tensor normal out unary impls
            pub(crate) mod normal_out_unary;
            /// a module contains cuda tensor random impls
            pub(crate) mod random;
            /// a module contains cuda tensor shape manipulation impls
            pub(crate) mod shape_manipulate;
            /// a module contains cuda tensor windows impls
            pub(crate) mod windows;
        }
        pub(crate) mod utils {
            pub(crate) mod reduce {
                pub(crate) mod reduce;
                pub(crate) mod reduce_template;
                pub(crate) mod reduce_utils;
            }
            pub(crate) mod binary {
                pub(crate) mod binary_normal;
            }
            pub(crate) mod unary {
                pub(crate) mod unary;
            }
            pub(crate) mod launch_cfg {
                pub(crate) mod launch_cfg_trait;
            }
        }
        /// a module contains cuda slice impls
        pub(crate) mod cuda_slice;
        /// a module contains cuda utils
        pub(crate) mod cuda_utils;
        /// a module contains cuda std ops impls
        pub(crate) mod std_ops;
    }

    /// a module contains all the common ops
    pub mod common {
        /// a module contains all the functions to help create a tensor
        pub mod creation;
        /// a module contains fast divmod ops
        pub mod divmod;
        /// a module contains reduce utils
        pub mod reduce;
        /// a module contains all the shape manipulation ops
        pub mod shape_manipulate;
        /// a module contains slice op
        pub mod slice;
    }
}

/// a module that wrap the _Tensor struct
pub mod tensor;
/// a module that defines the _Tensor struct
pub mod tensor_base;
/// a module that contains the implementation of the `Into` trait for the `_Tensor` struct.
///
/// # Note
/// for this library's developer, not necessary need to know how they works
pub mod to_tensor;
pub use crate::ops::cpu::utils::binary::binary_normal::binary_with_out;
use ctor::ctor;
pub use hpt_iterator::iterator_traits::*;
pub use hpt_iterator::TensorIterator;

pub use flate2;
// #[cfg(feature = "codegen")]
// pub use hpt_codegen::compile;
// #[cfg(feature = "codegen")]
// pub use hpt_codegen::fuse_proc_macro;
pub use hpt_allocator::resize_cpu_lru_cache;
#[cfg(feature = "cuda")]
mod cuda_exports {
    pub use hpt_allocator::resize_cuda_lru_cache;
    pub use hpt_allocator::Cuda;
}
#[cfg(feature = "cuda")]
pub use cuda_exports::*;
pub use hpt_allocator::{Backend, BackendTy, Buffer, Cpu};

pub use hpt_common::{error::base::TensorError, shape::shape::Shape, strides::strides::Strides};
pub use hpt_dataloader::data_loader::parse_header_compressed;
pub(crate) use hpt_dataloader::save;
pub use hpt_dataloader::{
    CompressionAlgo, DataLoader, Endian, FromSafeTensors, Load, MetaLoad, Save, TensorLoader,
    TensorSaver,
};
pub use hpt_macros::{select, Load, Save};
pub use hpt_traits::*;
pub use hpt_types::dtype::TypeCommon;
pub use hpt_types::into_scalar::Cast;
pub use hpt_types::into_vec::IntoVec;
pub use hpt_types::traits::VecTrait;
pub use hpt_types::type_promote::{
    BitWiseOut, Eval, FloatOutBinary, FloatOutBinaryPromote, FloatOutUnary, FloatOutUnaryPromote,
    NormalOut, NormalOutPromote, NormalOutUnary,
};
pub use hpt_types::vectors::*;
pub use serde;
pub use tensor::Tensor;

use std::{cell::RefCell, sync::atomic::AtomicUsize};
thread_local! {
    static THREAD_POOL: RefCell<threadpool::ThreadPool> = RefCell::new(
        threadpool::ThreadPool::new(num_cpus::get_physical())
    );
}

/// Set the Tensor display precision
pub fn set_display_precision(precision: usize) {
    DISPLAY_PRECISION.store(precision, std::sync::atomic::Ordering::Relaxed);
}

/// Set the left and right elements to display for each dimension
pub fn set_display_elements(lr_elements: usize) {
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
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    use hpt_cudakernels::RegisterInfo;
    use once_cell::sync::Lazy;

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

use hpt_types::arch_simd as simd;

#[cfg(target_feature = "avx2")]
type BoolVector = simd::_256bit::boolx32::boolx32;
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

#[cfg(feature = "cuda")]
const CUDA_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(2621654116416541);

#[allow(unused)]
/// Set the seed for random number generation
pub fn set_seed<B: BackendTy>(seed: u64) {
    match B::ID {
        0 => {
            panic!("CPU backend does not support setting seed");
        }
        #[cfg(feature = "cuda")]
        1 => {
            CUDA_SEED.store(seed, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {
            panic!("Unsupported backend {:?}", B::ID);
        }
    }
}
