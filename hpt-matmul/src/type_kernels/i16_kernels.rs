use crate::{ microkernel_trait::MatmulMicroKernel, I16Vec };
use num_traits::ConstZero;
use std::ops::Add;
use crate::type_kernels::common::avx2_kernels;

impl crate::Zero for i16 {
    const ZERO: Self = 0;
}

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel<I16Vec, i16, I16Vec> for i16 {
    fn get_kernel(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<Self>,
        crate::Pointer<Self>,
        crate::Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool
    ) {
        use crate::define_matmul_micro_kernel;
        use crate::define_neon_matmul_micro_kernel;
        assert_eq!(nr, 3);
        define_matmul_micro_kernel!(i16, I16Vec, x3x1, 3, 1);
        define_matmul_micro_kernel!(i16, I16Vec, x3x2, 3, 2);
        define_matmul_micro_kernel!(i16, I16Vec, x3x3, 3, 3);
        define_matmul_micro_kernel!(i16, I16Vec, x3x4, 3, 4);
        define_matmul_micro_kernel!(i16, I16Vec, x3x5, 3, 5);
        define_matmul_micro_kernel!(i16, I16Vec, x3x6, 3, 6);
        define_matmul_micro_kernel!(i16, I16Vec, x3x7, 3, 7);
        define_neon_matmul_micro_kernel!(i16, I16Vec, x3x8, 3, 8);
        [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8][mr - 1]
    }
    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(I16Vec, usize, usize) -> I16Vec
    >(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<Self>,
        crate::Pointer<Self>,
        crate::Pointer<Self>,
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
        G
    ) {
        use crate::define_post_op_matmul_micro_kernel;
        use crate::define_neon_post_op_matmul_micro_kernel;
        assert_eq!(nr, 3);
        define_post_op_matmul_micro_kernel!(i16, I16Vec, x3x1, 3, 1);
        define_post_op_matmul_micro_kernel!(i16, I16Vec, x3x2, 3, 2);
        define_post_op_matmul_micro_kernel!(i16, I16Vec, x3x3, 3, 3);
        define_post_op_matmul_micro_kernel!(i16, I16Vec, x3x4, 3, 4);
        define_post_op_matmul_micro_kernel!(i16, I16Vec, x3x5, 3, 5);
        define_post_op_matmul_micro_kernel!(i16, I16Vec, x3x6, 3, 6);
        define_post_op_matmul_micro_kernel!(i16, I16Vec, x3x7, 3, 7);
        define_neon_post_op_matmul_micro_kernel!(i16, I16Vec, x3x8, 3, 8);
        [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8][mr - 1]
    }
    fn get_max_mr() -> usize {
        8
    }
    fn get_max_nr() -> usize {
        3
    }

    fn get_horizontal_kernel(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<Self>,
        crate::Pointer<Self>,
        crate::Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool
    ) {
        assert_eq!(nr, 14);
        assert_eq!(mr, 1);
        use crate::define_neon_matmul_micro_kernel;
        define_neon_matmul_micro_kernel!(i16, I16Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_horizontal_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(I16Vec, usize, usize) -> I16Vec
    >(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<Self>,
        crate::Pointer<Self>,
        crate::Pointer<Self>,
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
        G
    ) {
        assert_eq!(nr, 14);
        assert_eq!(mr, 1);
        use crate::define_neon_post_op_matmul_micro_kernel;
        define_neon_post_op_matmul_micro_kernel!(i16, I16Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_horizontal_max_nr() -> usize {
        14
    }
}

#[cfg(target_feature = "avx2")]
avx2_kernels!(i16, I16Vec);