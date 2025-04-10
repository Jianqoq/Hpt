use crate::backends::cpu::kernels::conv2d::microkernel_trait::Conv2dMicroKernel;

#[cfg(target_feature = "neon")]
impl Conv2dMicroKernel for f32 {
}

#[cfg(not(target_feature = "neon"))]
impl Conv2dMicroKernel for f32 {}
