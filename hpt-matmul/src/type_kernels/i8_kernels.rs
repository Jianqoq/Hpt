use crate::{ microkernel_trait::MatmulMicroKernel, I8Vec };
use num_traits::ConstZero;
use std::ops::Add;
use crate::type_kernels::common::avx2_kernels;

impl crate::Zero for i8 {
    const ZERO: Self = 0;
}

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel<I8Vec, i8, I8Vec> for i8 {
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
        assert_eq!(nr, 1);
        define_matmul_micro_kernel!(i8, I8Vec, x1x1, 1, 1);
        define_matmul_micro_kernel!(i8, I8Vec, x1x2, 1, 2);
        define_matmul_micro_kernel!(i8, I8Vec, x1x3, 1, 3);
        define_matmul_micro_kernel!(i8, I8Vec, x1x4, 1, 4);
        define_matmul_micro_kernel!(i8, I8Vec, x1x5, 1, 5);
        define_matmul_micro_kernel!(i8, I8Vec, x1x6, 1, 6);
        define_matmul_micro_kernel!(i8, I8Vec, x1x7, 1, 7);
        define_matmul_micro_kernel!(i8, I8Vec, x1x8, 1, 8);
        define_matmul_micro_kernel!(i8, I8Vec, x1x9, 1, 9);
        define_matmul_micro_kernel!(i8, I8Vec, x1x10, 1, 10);
        define_matmul_micro_kernel!(i8, I8Vec, x1x11, 1, 11);
        define_matmul_micro_kernel!(i8, I8Vec, x1x12, 1, 12);
        define_matmul_micro_kernel!(i8, I8Vec, x1x13, 1, 13);
        define_matmul_micro_kernel!(i8, I8Vec, x1x14, 1, 14);
        define_matmul_micro_kernel!(i8, I8Vec, x1x15, 1, 15);
        define_neon_matmul_micro_kernel!(i8, I8Vec, x1x16, 1, 16);
        [
            x1x1,
            x1x2,
            x1x3,
            x1x4,
            x1x5,
            x1x6,
            x1x7,
            x1x8,
            x1x9,
            x1x10,
            x1x11,
            x1x12,
            x1x13,
            x1x14,
            x1x15,
            x1x16,
        ][mr - 1]
    }
    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(I8Vec, usize, usize) -> I8Vec
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
        use crate::define_neon_post_op_matmul_micro_kernel;
        use crate::define_post_op_matmul_micro_kernel;
        assert_eq!(nr, 1);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x1, 1, 1);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x2, 1, 2);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x3, 1, 3);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x4, 1, 4);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x5, 1, 5);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x6, 1, 6);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x7, 1, 7);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x8, 1, 8);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x9, 1, 9);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x10, 1, 10);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x11, 1, 11);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x12, 1, 12);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x13, 1, 13);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x14, 1, 14);
        define_post_op_matmul_micro_kernel!(i8, I8Vec, x1x15, 1, 15);
        define_neon_post_op_matmul_micro_kernel!(i8, I8Vec, x1x16, 1, 16);
        [
            x1x1,
            x1x2,
            x1x3,
            x1x4,
            x1x5,
            x1x6,
            x1x7,
            x1x8,
            x1x9,
            x1x10,
            x1x11,
            x1x12,
            x1x13,
            x1x14,
            x1x15,
            x1x16,
        ][mr - 1]
    }
    fn get_max_mr() -> usize {
        16
    }
    fn get_max_nr() -> usize {
        1
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
        define_neon_matmul_micro_kernel!(i8, I8Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_horizontal_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(I8Vec, usize, usize) -> I8Vec
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
        define_neon_post_op_matmul_micro_kernel!(i8, I8Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_horizontal_max_nr() -> usize {
        14
    }
}

#[cfg(target_feature = "avx2")]
avx2_kernels!(i8, I8Vec);