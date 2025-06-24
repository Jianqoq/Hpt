use crate::microkernel_trait::MatmulMicroKernel;
use crate::BoolVec;
use crate::Zero;
use crate::Add;
use crate::type_kernels::common::avx2_kernels;

impl Zero for bool {
    const ZERO: Self = false;
}

impl crate::Add for bool {
    #[inline(always)]
    fn add(self, other: Self) -> Self { self || other }
}

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel<BoolVec, bool, BoolVec> for bool {
    fn get_kernel(
        nr: usize,
        mr: usize,
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
    ) {
        use crate::define_matmul_micro_kernel;
        use crate::define_neon_matmul_micro_kernel;
        assert_eq!(nr, 1);
        define_matmul_micro_kernel!(bool, BoolVec, x1x1, 1, 1);
        define_matmul_micro_kernel!(bool, BoolVec, x1x2, 1, 2);
        define_matmul_micro_kernel!(bool, BoolVec, x1x3, 1, 3);
        define_matmul_micro_kernel!(bool, BoolVec, x1x4, 1, 4);
        define_matmul_micro_kernel!(bool, BoolVec, x1x5, 1, 5);
        define_matmul_micro_kernel!(bool, BoolVec, x1x6, 1, 6);
        define_matmul_micro_kernel!(bool, BoolVec, x1x7, 1, 7);
        define_matmul_micro_kernel!(bool, BoolVec, x1x8, 1, 8);
        define_matmul_micro_kernel!(bool, BoolVec, x1x9, 1, 9);
        define_matmul_micro_kernel!(bool, BoolVec, x1x10, 1, 10);
        define_matmul_micro_kernel!(bool, BoolVec, x1x11, 1, 11);
        define_matmul_micro_kernel!(bool, BoolVec, x1x12, 1, 12);
        define_matmul_micro_kernel!(bool, BoolVec, x1x13, 1, 13);
        define_matmul_micro_kernel!(bool, BoolVec, x1x14, 1, 14);
        define_matmul_micro_kernel!(bool, BoolVec, x1x15, 1, 15);
        define_neon_matmul_micro_kernel!(bool, BoolVec, x1x16, 1, 16);
        [
            x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8, x1x9, x1x10, x1x11, x1x12, x1x13,
            x1x14, x1x15, x1x16,
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
    ) -> fn(crate::Pointer<Self>, crate::Pointer<Self>, crate::Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        assert_eq!(mr, 1);
        assert_eq!(nr, 32);
        use crate::define_neon_matmul_micro_kernel;
        define_neon_matmul_micro_kernel!(bool, BoolVec, x32x1, 32, 1);
        x32x1
    }
    
    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(BoolVec, usize, usize) -> BoolVec
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
        assert_eq!(nr, 1);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x1, 1, 1);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x2, 1, 2);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x3, 1, 3);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x4, 1, 4);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x5, 1, 5);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x6, 1, 6);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x7, 1, 7);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x8, 1, 8);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x9, 1, 9);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x10, 1, 10);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x11, 1, 11);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x12, 1, 12);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x13, 1, 13);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x14, 1, 14);
        define_post_op_matmul_micro_kernel!(bool, BoolVec, x1x15, 1, 15);
        define_neon_post_op_matmul_micro_kernel!(bool, BoolVec, x1x16, 1, 16);
        [
            x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8, x1x9, x1x10, x1x11, x1x12, x1x13,
            x1x14, x1x15, x1x16,
        ][mr - 1]
    }
    
    fn get_horizontal_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(BoolVec, usize, usize) -> BoolVec
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
        assert_eq!(mr, 1);
        assert_eq!(nr, 32);
        use crate::define_neon_post_op_matmul_micro_kernel;
        define_neon_post_op_matmul_micro_kernel!(bool, BoolVec, x32x1, 32, 1);
        x32x1
    }
    
    fn get_horizontal_max_nr() -> usize {
        32
    }
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
avx2_kernels!(bool, BoolVec);

#[cfg(target_feature = "avx512f")]
avx2_kernels!(bool, BoolVec);