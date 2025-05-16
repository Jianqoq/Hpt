use crate::microkernel_trait::MatmulMicroKernel;
use crate::U64Vec;
use num_traits::ConstZero;
use std::ops::Add;

#[cfg(target_feature = "avx2")]
use crate::type_kernels::common::avx2_kernels;

impl crate::Zero for u64 {
    const ZERO: Self = 0;
}

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for u64 {
    type SelfVec = U64Vec;
    type MixedType = u64;
    type MixedVec = U64Vec;

    fn get_kernel(
        nr: usize,
        mr: usize
    ) -> fn(crate::Pointer<Self>, crate::Pointer<Self>, crate::Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        use crate::define_matmul_micro_kernel;
        use crate::define_neon_matmul_micro_kernel;
        assert_eq!(nr, 8);
        define_matmul_micro_kernel!(u64, U64Vec, x8x1, 8, 1);
        define_neon_matmul_micro_kernel!(u64, U64Vec, x8x2, 8, 2);
        [x8x1, x8x2][mr - 1]
    }

    fn get_horizontal_kernel(
        nr: usize,
        mr: usize
    ) -> fn(crate::Pointer<Self>, crate::Pointer<Self>, crate::Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        assert_eq!(nr, 14);
        assert_eq!(mr, 1);
        use crate::define_neon_matmul_micro_kernel;
        define_neon_matmul_micro_kernel!(u64, U64Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(U64Vec, usize, usize) -> U64Vec
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
        assert_eq!(nr, 8);
        define_post_op_matmul_micro_kernel!(u64, U64Vec, x8x1, 8, 1);
        define_neon_post_op_matmul_micro_kernel!(u64, U64Vec, x8x2, 8, 2);
        [x8x1, x8x2][mr - 1]
    }

    fn get_horizontal_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(U64Vec, usize, usize) -> U64Vec
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
        define_neon_post_op_matmul_micro_kernel!(u64, U64Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_max_mr() -> usize {
        2
    }

    fn get_max_nr() -> usize {
        8
    }

    fn get_horizontal_max_nr() -> usize {
        14
    }
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
avx2_kernels!(u64, U64Vec);

#[cfg(target_feature = "avx512f")]
avx512_kernels!(u64, U64Vec);