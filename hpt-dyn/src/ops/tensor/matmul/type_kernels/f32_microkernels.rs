use crate::ops::tensor::matmul::microkernel_trait::MatmulMicroKernel;

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for f32 {
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
        use crate::ops::tensor::matmul::microkernels::define_matmul_micro_kernel;
        use crate::ops::tensor::matmul::microkernels::define_neon_matmul_micro_kernel;
        assert_eq!(nr, 4);
        define_matmul_micro_kernel!(x4x1, 4, 1);
        define_matmul_micro_kernel!(x4x2, 4, 2);
        define_matmul_micro_kernel!(x4x3, 4, 3);
        define_neon_matmul_micro_kernel!(x4x4, 4, 4);
        [x4x1, x4x2, x4x3, x4x4][mr - 1]
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
        use crate::ops::tensor::matmul::microkernels::define_neon_post_op_matmul_micro_kernel;
        use crate::ops::tensor::matmul::microkernels::define_post_op_matmul_micro_kernel;
        assert_eq!(nr, 4);
        define_post_op_matmul_micro_kernel!(x4x1, 4, 1);
        define_post_op_matmul_micro_kernel!(x4x2, 4, 2);
        define_post_op_matmul_micro_kernel!(x4x3, 4, 3);
        define_neon_post_op_matmul_micro_kernel!(x4x4, 4, 4);
        [x4x1, x4x2, x4x3, x4x4][mr - 1]
    }
    fn get_max_mr() -> usize {
        4
    }
    fn get_max_nr() -> usize {
        4
    }
}

#[cfg(target_feature = "avx2")]
impl MatmulMicroKernel for f32 {
    fn get_inline_asm_kernel(
        nr: usize,
        mr: usize,
        has_rem: bool,
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
        assert_eq!(nr, 2);
        use crate::ops::tensor::matmul::microkernels::define_matmul_micro_kernel_inline_asm;
        use crate::ops::tensor::matmul::microkernels::define_matmul_micro_kernel_inline_asm_rem;

        define_matmul_micro_kernel_inline_asm!(
            x2x1,
            2,
            1,
            8,
            vxorps,
            vmovaps,
            vmovups,
            vbroadcastss,
            vfmadd231ps,
            vaddps,
            [[ymm0, ymm1],],
            [ymm2, ymm3],
            [ymm4],
            [ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15]
        );

        define_matmul_micro_kernel_inline_asm!(
            x2x2,
            2,
            2,
            8,
            vxorps,
            vmovaps,
            vmovups,
            vbroadcastss,
            vfmadd231ps,
            vaddps,
            [[ymm0, ymm1], [ymm2, ymm3],],
            [ymm4, ymm5],
            [ymm6, ymm7],
            [ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15]
        );

        define_matmul_micro_kernel_inline_asm!(
            x2x3,
            2,
            3,
            8,
            vxorps,
            vmovaps,
            vmovups,
            vbroadcastss,
            vfmadd231ps,
            vaddps,
            [[ymm0, ymm1], [ymm2, ymm3], [ymm4, ymm5],],
            [ymm6, ymm7],
            [ymm8, ymm9, ymm10],
            [ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15]
        );

        define_matmul_micro_kernel_inline_asm!(
            x2x4,
            2,
            4,
            8,
            vxorps,
            vmovaps,
            vmovups,
            vbroadcastss,
            vfmadd231ps,
            vaddps,
            [[ymm0, ymm1], [ymm2, ymm3], [ymm4, ymm5], [ymm6, ymm7],],
            [ymm8, ymm9],
            [ymm10, ymm11, ymm12, ymm13],
            [ymm10, ymm11, ymm12, ymm13, ymm14, ymm15]
        );

        define_matmul_micro_kernel_inline_asm!(
            x2x5,
            2,
            5,
            8,
            vxorps,
            vmovaps,
            vmovups,
            vbroadcastss,
            vfmadd231ps,
            vaddps,
            [
                [ymm0, ymm1],
                [ymm2, ymm3],
                [ymm4, ymm5],
                [ymm6, ymm7],
                [ymm8, ymm9],
            ],
            [ymm10, ymm11],
            [ymm12, ymm13, ymm14, ymm15],
            [ymm12, ymm13, ymm14, ymm15]
        );

        define_matmul_micro_kernel_inline_asm!(
            x2x6,
            2,
            6,
            8,
            vxorps,
            vmovaps,
            vmovups,
            vbroadcastss,
            vfmadd231ps,
            vaddps,
            [
                [ymm0, ymm1],
                [ymm2, ymm3],
                [ymm4, ymm5],
                [ymm6, ymm7],
                [ymm8, ymm9],
                [ymm10, ymm11],
            ],
            [ymm12, ymm13],
            [ymm14, ymm15],
            [ymm14, ymm15]
        );
        define_matmul_micro_kernel_inline_asm_rem!(x2x1_rem, 2, 1);
        define_matmul_micro_kernel_inline_asm_rem!(x2x2_rem, 2, 2);
        define_matmul_micro_kernel_inline_asm_rem!(x2x3_rem, 2, 3);
        define_matmul_micro_kernel_inline_asm_rem!(x2x4_rem, 2, 4);
        define_matmul_micro_kernel_inline_asm_rem!(x2x5_rem, 2, 5);
        define_matmul_micro_kernel_inline_asm_rem!(x2x6_rem, 2, 6);

        [
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6],
            [x2x1_rem, x2x2_rem, x2x3_rem, x2x4_rem, x2x5_rem, x2x6_rem],
        ][has_rem as usize][mr - 1]
    }

    fn get_max_mr() -> usize {
        6
    }
}
