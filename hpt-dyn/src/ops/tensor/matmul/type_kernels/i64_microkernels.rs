use crate::ops::tensor::matmul::microkernel_trait::MatmulMicroKernel;

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for i64 {
    fn get_kernel(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<Self>,
        hpt_common::Pointer<Self>,
        hpt_common::Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
    ) {
        assert_eq!(nr, 8);
        use crate::ops::tensor::matmul::microkernels::define_matmul_micro_kernel;
        use crate::ops::tensor::matmul::microkernels::define_neon_matmul_micro_kernel;
        define_matmul_micro_kernel!(x8x1, 8, 1);
        define_neon_matmul_micro_kernel!(x8x2, 8, 2);
        [x8x1, x8x2][mr - 1]
    }
    fn get_kernel_with_post_op<F: Fn(Self, usize, usize) -> Self, G: Fn(Self::Vec, usize, usize) -> Self::Vec>(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<Self>,
        hpt_common::Pointer<Self>,
        hpt_common::Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        bool,
        usize,
        usize,
        F,
        G,
    ) {
        use crate::ops::tensor::matmul::microkernels::define_post_op_matmul_micro_kernel;
        use crate::ops::tensor::matmul::microkernels::define_neon_post_op_matmul_micro_kernel;
        assert_eq!(nr, 8);
        define_post_op_matmul_micro_kernel!(x8x1, 8, 1);
        define_neon_post_op_matmul_micro_kernel!(x8x2, 8, 2);
        [x8x1, x8x2][mr - 1]
    }
    fn get_max_mr() -> usize {
        2
    }
    fn get_max_nr() -> usize {
        8
    }
}

#[cfg(not(target_feature = "neon"))]
impl MatmulMicroKernel for i64 {}
