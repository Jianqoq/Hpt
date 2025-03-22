use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for u16 {
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
        use crate::define_matmul_micro_kernel;
        use crate::define_neon_matmul_micro_kernel;
        assert_eq!(nr, 3);
        define_matmul_micro_kernel!(x3x1, 3, 1);
        define_matmul_micro_kernel!(x3x2, 3, 2);
        define_matmul_micro_kernel!(x3x3, 3, 3);
        define_matmul_micro_kernel!(x3x4, 3, 4);
        define_matmul_micro_kernel!(x3x5, 3, 5);
        define_matmul_micro_kernel!(x3x6, 3, 6);
        define_matmul_micro_kernel!(x3x7, 3, 7);
        define_neon_matmul_micro_kernel!(x3x8, 3, 8);
        [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8][mr - 1]
    }
    fn get_kernel_with_post_op<F: Fn(Self) -> Self, G: Fn(Self::Vec) -> Self::Vec>(
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
        F,
        G,
    ) {
        use crate::define_neon_post_op_matmul_micro_kernel;
        use crate::define_post_op_matmul_micro_kernel;
        assert_eq!(nr, 3);
        define_post_op_matmul_micro_kernel!(x3x1, 3, 1);
        define_post_op_matmul_micro_kernel!(x3x2, 3, 2);
        define_post_op_matmul_micro_kernel!(x3x3, 3, 3);
        define_post_op_matmul_micro_kernel!(x3x4, 3, 4);
        define_post_op_matmul_micro_kernel!(x3x5, 3, 5);
        define_post_op_matmul_micro_kernel!(x3x6, 3, 6);
        define_post_op_matmul_micro_kernel!(x3x7, 3, 7);
        define_neon_post_op_matmul_micro_kernel!(x3x8, 3, 8);
        [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8][mr - 1]
    }
    fn get_max_mr() -> usize {
        8
    }
    fn get_max_nr() -> usize {
        3
    }
}

#[cfg(not(target_feature = "neon"))]
impl MatmulMicroKernel for u16 {}
