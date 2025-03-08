//! This crate is dynamic graph based tensor library
#![deny(missing_docs)]

/// a module contains all the Tensor operations. include the CPU and GPU operations
pub(crate) mod backends {
    /// a module contains all the CPU operations
    pub(crate) mod cpu {
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
        pub(crate) mod std_ops;
        /// a module defines all the kernels
        pub(crate) mod kernels {
            /// a module defines reduce kernels
            pub(crate) mod argreduce_kernels;
            /// a module defines the batchnorm conv2d kernels
            pub(crate) mod batch_norm_conv;
            /// a module defines the conv2d kernels
            pub(crate) mod conv;
            /// a module defines the dwconv2d kernels
            pub(crate) mod conv_group;
            /// a module defines the conv transpose kernels
            pub(crate) mod conv_transpose;
            /// a module defines the dwconv2d kernels
            pub(crate) mod dwconv;
            /// a module defines the lp_pool2d kernels
            pub(crate) mod lp_pool_kernels;
            /// a module defines the reduce kernels
            pub(crate) mod reduce;
            /// a module defines the softmax kernels
            pub(crate) mod softmax;
            /// a module contains all the pooling operations
            pub(crate) mod pooling {
                /// a module contains all the common pooling operations
                pub(crate) mod common;
            }
            /// a module defines normalization operations
            pub(crate) mod normalization {
                /// a module defines log_softmax
                pub(crate) mod log_softmax;
                /// a module defines the logsoftmax kernels
                pub(crate) mod logsoftmax;
                /// a module defines softmax utils
                pub(crate) mod normalize_utils;
                /// a module defines softmax
                pub(crate) mod softmax;
            }
            /// a module defines conv2d operation
            pub(crate) mod conv2d {
                /// a module defines batchnorm_conv2d operation
                pub(crate) mod batchnorm_conv2d;
                /// a module defines conv2d operation
                pub(crate) mod conv2d;
                /// a module defines conv2d_group operation
                pub(crate) mod conv2d_group;
                /// a module defines conv2d_transpose operation
                pub(crate) mod conv2d_transpose;
                /// a module defines dwconv2d operation
                pub(crate) mod dwconv2d;
            }
        }
        /// a module that contains all the functions expose for the external user (we may have diff tensor (differentiable tensor) in the future)
        pub(crate) mod tensor_external {
            /// a module that contains all the advance operations
            pub(crate) mod advance;
            /// a module that contains all the arg reduce functions
            pub(crate) mod arg_reduce;
            /// a module defines all normal binary operation
            pub(crate) mod binary;
            /// a module that contains all the tensor compare functions
            pub(crate) mod cmp;
            /// a module that contains all the common reduce functions
            pub(crate) mod common_reduce;
            /// a module that contains all the conv functions
            pub(crate) mod conv;
            /// a module that contains all the cumulative operations
            pub(crate) mod cumulative;
            /// a module that contains all fft operations
            pub(crate) mod fft;
            /// a module that contains all the float out binary operations
            pub(crate) mod float_out_binary;
            /// a module that contains all the unary operations that has floating type output
            pub(crate) mod float_out_unary;
            /// a module that contains matrix multiplication operations
            pub(crate) mod matmul;
            /// a module that contains all normal methods to create a tensor
            pub(crate) mod normal_creation;
            /// a module that contains all the unary operations that has self type output
            pub(crate) mod normal_out_unary;
            /// a module that contains all the normalization functions
            pub(crate) mod normalization;
            /// a module that contains all the pooling functions
            pub(crate) mod pooling;
            /// a module that contains all the random number generate functions
            pub(crate) mod random;
            /// a module that contains all the regularization functions
            pub(crate) mod regularization;
            /// a module that contains all the shape manipulation functions
            pub(crate) mod shape_manipulate;
            /// a module that contains all the slice functions
            pub(crate) mod slice;
            /// a module that contains all the tensordot functions
            pub(crate) mod tensordot;
            /// a module that contains all the windows creation functions
            pub(crate) mod windows;
        }
        /// a module that contains all the functions only for the internal user (we may have diff tensor (differentiable tensor) in the future)
        pub(crate) mod tensor_internal {
            /// a module that contains all the advance operations
            pub(crate) mod advance;
            /// a module that contains all the arg reduce functions
            pub(crate) mod arg_reduce;
            /// a module that contains all the tensor compare functions
            pub(crate) mod cmp;
            /// a module that contains all the common reduce functions
            pub(crate) mod common_reduce;
            /// a module that contains all the conv functions
            pub(crate) mod conv;
            /// a module that contains all the cumulative operations
            pub(crate) mod cumulative;
            /// a module that contains all fft operations
            pub(crate) mod fft;
            /// a module that contains all the float out binary operations
            pub(crate) mod float_out_binary;
            /// a module that contains all the unary operations that has floating type output
            pub(crate) mod float_out_unary;
            /// a module that contains matrix multiplication operations
            pub(crate) mod matmul;
            /// a module that contains all normal methods to create a tensor
            pub(crate) mod normal_creation;
            /// a module that contains all the unary operations that has self type output
            pub(crate) mod normal_out_unary;
            /// a module that contains all the normalization functions
            pub(crate) mod normalization;
            /// a module that contains all the pooling functions
            pub(crate) mod pooling;
            /// a module that contains all the random number generate functions
            pub(crate) mod random;
            /// a module that contains all the regularization functions
            pub(crate) mod regularization;
            /// a module that contains all the shape manipulation functions
            pub(crate) mod shape_manipulate;
            /// a module that contains all the tensordot functions
            pub(crate) mod tensordot;
            /// a module that contains all the windows creation functions
            pub(crate) mod windows;
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
        pub(crate) mod tensor_external {
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
    pub(crate) mod common {
        /// a module contains all the functions to help create a tensor
        pub(crate) mod creation;
        /// a module contains fast divmod ops
        pub(crate) mod divmod;
        /// a module contains reduce utils
        pub(crate) mod reduce;
        /// a module contains all the shape manipulation ops
        pub(crate) mod shape_manipulate;
        /// a module contains slice op
        pub(crate) mod slice;
    }
}

pub(crate) mod tensor;
pub(crate) mod tensor_base;
pub(crate) mod to_tensor;

use std::{cell::RefCell, sync::atomic::AtomicUsize};
thread_local! {
    static THREAD_POOL: RefCell<threadpool::ThreadPool> = RefCell::new(
        threadpool::ThreadPool::new(num_cpus::get_physical())
    );
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

use ctor::ctor;
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
    target_feature = "neon"
))]
type BoolVector = simd::_128bit::boolx16::boolx16;

use hpt_types::traits::VecTrait;
const SIMD_WIDTH: usize =
    <f32 as hpt_types::dtype::TypeCommon>::Vec::SIZE * std::mem::size_of::<f32>() * 8;

#[cfg(feature = "cuda")]
const CUDA_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(2621654116416541);

/// this module contains all the operators for the Tensor
pub mod ops {
    pub use hpt_traits::ops::advance::*;
    pub use hpt_traits::ops::binary::*;
    pub use hpt_traits::ops::cmp::*;
    pub use hpt_traits::ops::conv::*;
    pub use hpt_traits::ops::creation::*;
    pub use hpt_traits::ops::cumulative::*;
    pub use hpt_traits::ops::fft::*;
    pub use hpt_traits::ops::normalization::*;
    pub use hpt_traits::ops::pooling::*;
    pub use hpt_traits::ops::random::*;
    pub use hpt_traits::ops::reduce::*;
    pub use hpt_traits::ops::regularization::*;
    pub use hpt_traits::ops::shape_manipulate::*;
    pub use hpt_traits::ops::slice::*;
    pub use hpt_traits::ops::unary::*;
    pub use hpt_traits::ops::windows::*;
}

/// module for error handling
pub mod error {
    pub use hpt_common::error::base::TensorError;
}

/// module for common utils like shape and strides
pub mod common {
    pub use hpt_common::{shape::shape::Shape, strides::strides::Strides};
    pub use hpt_traits::tensor::{CommonBounds, TensorInfo};
    /// common utils for cpu
    pub mod cpu {
        pub use hpt_traits::tensor::TensorLike;
    }
}

/// module for memory allocation
pub mod alloc {
    pub use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
}

/// module for tensor iterator
pub mod iter {
    pub use hpt_iterator::iterator_traits::*;
    pub use hpt_iterator::TensorIterator;
}

/// type related module
pub mod types {
    pub use half::{bf16, f16};
    /// module contains vector types and traits
    pub mod vectors {
        pub use hpt_types::vectors::*;
        /// module contains vector traits
        pub mod traits {
            pub use hpt_types::traits::VecTrait;
        }
    }
    /// module contains cast traits, perform type conversion
    pub mod cast {
        pub use hpt_types::into_scalar::Cast;
        pub use hpt_types::into_vec::IntoVec;
    }
    /// module contains math traits for scalar and vector, all the methods will auto promote the type
    pub mod math {
        pub use hpt_types::type_promote::{
            BitWiseOut, Eval, FloatOutBinary, FloatOutBinaryPromote, FloatOutUnary,
            FloatOutUnaryPromote, NormalOut, NormalOutPromote, NormalOutUnary,
        };
    }
    /// module contains type common traits
    pub use hpt_types::dtype::TypeCommon;
}

/// reexport serde
pub mod serialize {
    pub use serde;
}

pub use hpt_dataloader::{Load, Save};
pub use hpt_macros::{Load, Save};

/// module for save and load
pub mod save_load {
    pub use flate2;
    pub use hpt_dataloader::data_loader::parse_header_compressed;
    pub use hpt_dataloader::{
        save, CompressionAlgo, DataLoader, Endian, FromSafeTensors, MetaLoad, TensorLoader,
        TensorSaver,
    };
}

/// module for backend
pub mod backend {
    pub use hpt_allocator::Cpu;
    #[cfg(feature = "cuda")]
    pub use hpt_allocator::Cuda;

    pub use hpt_allocator::{BackendTy, Buffer};
}

/// module for buitin templates
pub mod buitin_templates {
    /// module for cpu buitin templates
    pub mod cpu {
        pub use crate::backends::cpu::utils::binary::binary_normal::binary_with_out;
    }
}

/// module for utils, like set_num_threads, set_seed, etc.
pub mod utils {
    #[cfg(feature = "cuda")]
    use crate::CUDA_SEED;
    use crate::{DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION, THREAD_POOL};
    pub use hpt_allocator::resize_cpu_lru_cache;
    #[cfg(feature = "cuda")]
    pub use hpt_allocator::resize_cuda_lru_cache;
    pub use hpt_macros::select;

    /// Get the global number of threads
    pub fn get_num_threads() -> usize {
        THREAD_POOL.with(|x| x.borrow().max_count())
    }
    /// Set the Tensor display precision
    pub fn set_display_precision(precision: usize) {
        DISPLAY_PRECISION.store(precision, std::sync::atomic::Ordering::Relaxed);
    }
    /// Set the left and right elements to display for each dimension
    pub fn set_display_elements(lr_elements: usize) {
        DISPLAY_LR_ELEMENTS.store(lr_elements, std::sync::atomic::Ordering::Relaxed);
    }
    #[allow(unused)]
    /// Set the seed for random number generation
    pub fn set_seed<B: crate::backend::BackendTy>(seed: u64) {
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
}

pub use tensor::Tensor;
