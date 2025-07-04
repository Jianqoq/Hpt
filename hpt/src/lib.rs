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
            pub(crate) mod argreduce_kernels;
            pub(crate) mod reduce;
            pub(crate) mod softmax;
            pub(crate) mod pooling {
                pub(crate) mod common;
            }
            pub(crate) mod normalization {
                pub(crate) mod batch_norm;
                pub(crate) mod log_softmax;
                pub(crate) mod logsoftmax;
                pub(crate) mod normalize_utils;
                pub(crate) mod softmax;
            }
            pub(crate) mod conv2d {
                pub(crate) mod batchnorm_conv2d;
                pub(crate) mod conv2d;
                pub(crate) mod conv2d_direct;
                pub(crate) mod conv2d_group;
                pub(crate) mod dwconv2d;
                pub(crate) mod type_kernels {
                    pub(crate) mod bf16_microkernels;
                    pub(crate) mod bool_microkernels;
                    pub(crate) mod complex32_microkernels;
                    pub(crate) mod complex64_microkernels;
                    pub(crate) mod f16_microkernels;
                    pub(crate) mod f32_microkernels;
                    pub(crate) mod f64_microkernels;
                    pub(crate) mod i16_microkernels;
                    pub(crate) mod i32_microkernels;
                    pub(crate) mod i64_microkernels;
                    pub(crate) mod i8_microkernels;
                    pub(crate) mod isize_microkernels;
                    pub(crate) mod u16_microkernels;
                    pub(crate) mod u32_microkernels;
                    pub(crate) mod u64_microkernels;
                    pub(crate) mod u8_microkernels;
                    pub(crate) mod usize_microkernels;
                }
                pub(crate) mod conv2d_img2col;
                pub(crate) mod conv2d_micro_kernels;
                pub(crate) mod conv2d_new_mp;
                pub(crate) mod microkernel_trait;
                pub(crate) mod utils;
            }
            /// a module defines gemm operation for cpu
            pub(crate) mod matmul {
                pub(crate) mod common;
                pub(crate) mod matmul;
                pub(crate) mod template;
                pub(crate) mod matmul_mixed_precision;
                pub(crate) mod matmul_mp_post;
                pub(crate) mod matmul_post;
                pub(crate) mod microkernel_trait;
                pub(crate) mod microkernels;
                pub(crate) mod utils;
                pub(crate) mod type_kernels {
                    pub(crate) mod bf16_microkernels;
                    pub(crate) mod bool_microkernels;
                    pub(crate) mod complex32_microkernels;
                    pub(crate) mod complex64_microkernels;
                    pub(crate) mod f16_microkernels;
                    pub(crate) mod f32_microkernels;
                    pub(crate) mod f64_microkernels;
                    pub(crate) mod i16_microkernels;
                    pub(crate) mod i32_microkernels;
                    pub(crate) mod i64_microkernels;
                    pub(crate) mod i8_microkernels;
                    pub(crate) mod isize_microkernels;
                    pub(crate) mod u16_microkernels;
                    pub(crate) mod u32_microkernels;
                    pub(crate) mod u64_microkernels;
                    pub(crate) mod u8_microkernels;
                    pub(crate) mod usize_microkernels;
                }
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
            /// a module that contains all the gemm functions
            pub(crate) mod gemm;
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
            /// a module that contains all the gemm functions
            pub(crate) mod gemm;
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
            /// a module contains cuda tensor conv impls
            pub(crate) mod conv2d;
            /// a module contains cuda tensor float out binary impls
            pub(crate) mod float_out_binary;
            /// a module contains cuda tensor float out unary impls
            pub(crate) mod float_out_unary;
            /// a module contains cuda tensor layernorm impls
            pub(crate) mod layernorm;
            /// a module contains cuda matmul impls
            pub(crate) mod matmul;
            /// a module contains cuda tensor normal creation impls
            pub(crate) mod normal_creation;
            /// a module contains cuda tensor normal out unary impls
            pub(crate) mod normal_out_unary;
            /// a module contains cuda tensor normalization impls
            pub(crate) mod normalization;
            /// a module contains cuda tensor pooling impls
            pub(crate) mod pooling;
            /// a module contains cuda tensor shape manipulation impls
            pub(crate) mod shape_manipulate;
            /// a module contains cuda tensor softmax impls
            pub(crate) mod softmax;
            /// a module contains cuda tensor windows impls
            pub(crate) mod windows;
        }
        pub(crate) mod tensor_external {
            /// a module contains cuda tensor arg reduce impls
            pub(crate) mod arg_reduce;
            /// a module that contains inplace binary operations
            pub(crate) mod binary;
            /// a module contains cuda tensor cmp impls
            pub(crate) mod cmp;
            /// a module contains cuda tensor common reduce impls
            pub(crate) mod common_reduce;
            /// a module contains cuda tensor conv2d impls
            pub(crate) mod conv2d;
            /// a module contains cuda tensor float out binary impls
            pub(crate) mod float_out_binary;
            /// a module contains cuda tensor float out unary impls
            pub(crate) mod float_out_unary;
            /// a module contains cuda tensor gemm impls
            pub(crate) mod gemm;
            /// a module contains cuda tensor matmul impls
            pub(crate) mod matmul;
            /// a module contains cuda tensor normal creation impls
            pub(crate) mod normal_creation;
            /// a module contains cuda tensor normal out unary impls
            pub(crate) mod normal_out_unary;
            /// a module contains cuda tensor normalization impls
            pub(crate) mod normalization;
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
        /// a module contains conv utils
        pub(crate) mod conv;
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
        pub(crate) mod thread_pool;
        pub(crate) mod prefetch;
    }
}
pub(crate) mod tensor;
pub(crate) mod tensor_base;
pub(crate) mod to_tensor;
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
    pub use hpt_common::{shape::shape::Shape, strides::strides::Strides, Pointer};
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
    pub use rayon;
}

/// type related module
pub mod types {
    pub use half::{bf16, f16};
    pub use num::complex::{Complex32, Complex64};
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
pub mod re_exports {
    #[cfg(feature = "cuda")]
    pub use cudarc;
    pub use seq_macro;
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
    use crate::{
        CUSTOM_THREAD_POOL, DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION, RAYON_NUM_THREADS, RAYON_POOL,
        THREAD_POOL,
    };
    pub use hpt_allocator::resize_cpu_lru_cache;
    #[cfg(feature = "cuda")]
    pub use hpt_allocator::resize_cuda_lru_cache;
    pub use hpt_macros::select;

    /// Get the global number of threads
    pub fn get_num_threads() -> usize {
        CUSTOM_THREAD_POOL.with(|x| x.borrow().num_threads())
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
            Ok(_) => {
                RAYON_NUM_THREADS.store(num_threads, std::sync::atomic::Ordering::Relaxed);
                RAYON_POOL.with(|x| {
                    let mut x = x.borrow_mut();
                    *x = rayon::ThreadPoolBuilder::new()
                        .num_threads(num_threads)
                        .stack_size(4 * 1024 * 1024)
                        .build()
                        .unwrap();
                });
                CUSTOM_THREAD_POOL.with(|x| {
                    x.borrow_mut().resize(num_threads).unwrap();
                });
            }
            Err(_) => {}
        }
    }
}

use ctor::ctor;
use hpt_types::arch_simd as simd;
use once_cell::sync::Lazy;
use std::{cell::RefCell, sync::atomic::AtomicUsize};
pub use tensor::Tensor;

#[ctor]
fn init() {
    THREAD_POOL.with(|x| {
        x.borrow_mut().set_num_threads(num_cpus::get_physical());
    });
    RAYON_POOL.with(|x| {
        let mut x = x.borrow_mut();
        *x = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get_physical())
            .stack_size(4 * 1024 * 1024)
            .build()
            .unwrap();
    });
    CUSTOM_THREAD_POOL.with(|x| {
        x.borrow_mut().resize(num_cpus::get_physical()).unwrap();
    });
}

thread_local! {
    static THREAD_POOL: RefCell<threadpool::ThreadPool> = RefCell::new(
        threadpool::ThreadPool::new(num_cpus::get_physical())
    );
}

thread_local! {
    static CUSTOM_THREAD_POOL: RefCell<crate::backends::common::thread_pool::ComputeThreadPool> = RefCell::new(
        crate::backends::common::thread_pool::ComputeThreadPool::new(num_cpus::get_physical())
    );
}

thread_local! {
    static RAYON_POOL: RefCell<rayon::ThreadPool> = RefCell::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get_physical())
            .build()
            .unwrap()
    );
}

static DISPLAY_PRECISION: AtomicUsize = AtomicUsize::new(4);
static DISPLAY_LR_ELEMENTS: AtomicUsize = AtomicUsize::new(3);
static ALIGN: usize = 64;
static RAYON_NUM_THREADS: Lazy<AtomicUsize> = Lazy::new(|| AtomicUsize::new(num_cpus::get()));

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
type BoolVector = simd::_256bit::boolx32;
#[cfg(target_feature = "avx512f")]
type BoolVector = simd::_512bit::boolx64;
#[cfg(any(
    all(not(target_feature = "avx2"), target_feature = "sse"),
    target_feature = "neon"
))]
type BoolVector = simd::_128bit::boolx16;

#[cfg(feature = "cuda")]
const CUDA_SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(2621654116416541);

#[cfg(feature = "cuda")]
thread_local! {
    static CUDNN: RefCell<
        std::collections::HashMap<usize, std::sync::Arc<cudarc::cudnn::Cudnn>>
    > = std::collections::HashMap::new().into();
}

#[cfg(feature = "cuda")]
thread_local! {
    static CUBLAS: RefCell<
        std::collections::HashMap<usize, std::sync::Arc<cudarc::cublas::CudaBlas>>
    > = std::collections::HashMap::new().into();
}
