use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for isize {
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
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(nr, 8);
            define_matmul_micro_kernel!(x8x1, 8, 1);
            define_neon_matmul_micro_kernel!(x8x2, 8, 2);
            [x8x1, x8x2][mr - 1]
        }
        #[cfg(target_pointer_width = "32")]
        {
            assert_eq!(nr, 4);
            define_matmul_micro_kernel!(x4x1, 4, 1);
            define_matmul_micro_kernel!(x4x2, 4, 2);
            define_matmul_micro_kernel!(x4x3, 4, 3);
            define_neon_matmul_micro_kernel!(x4x4, 4, 4);
            [x4x1, x4x2, x4x3, x4x4][mr - 1]
        }
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
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(nr, 8);
            define_post_op_matmul_micro_kernel!(x8x1, 8, 1);
            define_neon_post_op_matmul_micro_kernel!(x8x2, 8, 2);
            [x8x1, x8x2][mr - 1]
        }
        #[cfg(target_pointer_width = "32")]
        {
            assert_eq!(nr, 4);
            define_post_op_matmul_micro_kernel!(x4x1, 4, 1);
            define_post_op_matmul_micro_kernel!(x4x2, 4, 2);
            define_post_op_matmul_micro_kernel!(x4x3, 4, 3);
            define_neon_post_op_matmul_micro_kernel!(x4x4, 4, 4);
            [x4x1, x4x2, x4x3, x4x4][mr - 1]
        }
    }
    fn get_max_mr() -> usize {
        #[cfg(target_pointer_width = "64")]
        {
            2
        }
        #[cfg(target_pointer_width = "32")]
        {
            4
        }
    }
    fn get_max_nr() -> usize {
        #[cfg(target_pointer_width = "64")]
        {
            8
        }
        #[cfg(target_pointer_width = "32")]
        {
            4
        }
    }
}

#[cfg(not(target_feature = "neon"))]
impl MatmulMicroKernel for isize {}
