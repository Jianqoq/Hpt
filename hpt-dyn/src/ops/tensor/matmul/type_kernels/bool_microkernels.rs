use crate::ops::tensor::matmul::microkernel_trait::MatmulMicroKernel;

#[cfg(not(target_feature = "neon"))]
impl MatmulMicroKernel for bool {}
