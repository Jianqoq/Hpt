use crate::microkernel_trait::MatmulMicroKernel;
use crate::I32Vec;
use num_traits::ConstZero;
use std::ops::Add;

#[cfg(target_feature = "avx2")]
use crate::type_kernels::common::avx2_kernels;

impl crate::Zero for i32 {
    const ZERO: Self = 0;
}

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for i32 {
    type SelfVec = I32Vec;
    type MixedType = i32;
    type MixedVec = I32Vec;

    fn get_kernel(
        nr: usize,
        mr: usize
    ) -> fn(crate::Pointer<Self>, crate::Pointer<Self>, crate::Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        use crate::define_matmul_micro_kernel;
        use crate::define_neon_matmul_micro_kernel;
        assert_eq!(nr, 4);
        define_matmul_micro_kernel!(i32, I32Vec, x4x1, 4, 1);
        define_matmul_micro_kernel!(i32, I32Vec, x4x2, 4, 2);
        define_matmul_micro_kernel!(i32, I32Vec, x4x3, 4, 3);
        define_neon_matmul_micro_kernel!(i32, I32Vec, x4x4, 4, 4);
        [x4x1, x4x2, x4x3, x4x4][mr - 1]
    }

    fn get_horizontal_kernel(
        nr: usize,
        mr: usize
    ) -> fn(crate::Pointer<Self>, crate::Pointer<Self>, crate::Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        assert_eq!(nr, 14);
        assert_eq!(mr, 1);
        use crate::define_neon_matmul_micro_kernel;
        define_neon_matmul_micro_kernel!(i32, I32Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(I32Vec, usize, usize) -> I32Vec
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
        assert_eq!(nr, 4);
        define_post_op_matmul_micro_kernel!(i32, I32Vec, x4x1, 4, 1);
        define_post_op_matmul_micro_kernel!(i32, I32Vec, x4x2, 4, 2);
        define_post_op_matmul_micro_kernel!(i32, I32Vec, x4x3, 4, 3);
        define_neon_post_op_matmul_micro_kernel!(i32, I32Vec, x4x4, 4, 4);
        [x4x1, x4x2, x4x3, x4x4][mr - 1]
    }

    fn get_horizontal_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(I32Vec, usize, usize) -> I32Vec
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
        define_neon_post_op_matmul_micro_kernel!(i32, I32Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_max_mr() -> usize {
        4
    }

    fn get_max_nr() -> usize {
        4
    }

    fn get_horizontal_max_nr() -> usize {
        14
    }
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
avx2_kernels!(i32, I32Vec);

#[cfg(target_feature = "avx512f")]
avx512_kernels!(i32, I32Vec);