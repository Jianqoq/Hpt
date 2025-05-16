use crate::microkernel_trait::MatmulMicroKernel;
use crate::Cplx64Vec;
use num_complex::Complex64;
use std::ops::Add;
#[cfg(target_feature = "avx2")]
use crate::type_kernels::common::avx2_kernels;

impl crate::Zero for Complex64 {
    const ZERO: Self = Complex64::new(0.0, 0.0);
}

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for Complex64 {
    type SelfVec = Cplx64Vec;
    type MixedType = Complex64;
    type MixedVec = Cplx64Vec;

    fn get_kernel(
        nr: usize,
        mr: usize
    ) -> fn(crate::Pointer<Self>, crate::Pointer<Self>, crate::Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        use crate::define_matmul_micro_kernel;
        use crate::define_neon_matmul_micro_kernel;
        assert_eq!(nr, 8);
        define_matmul_micro_kernel!(Complex64, Cplx64Vec, x8x1, 8, 1);
        define_neon_matmul_micro_kernel!(Complex64, Cplx64Vec, x8x2, 8, 2);
        [x8x1, x8x2][mr - 1]
    }

    fn get_horizontal_kernel(
        nr: usize,
        mr: usize
    ) -> fn(crate::Pointer<Self>, crate::Pointer<Self>, crate::Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        assert_eq!(nr, 14);
        assert_eq!(mr, 1);
        use crate::define_neon_matmul_micro_kernel;
        define_neon_matmul_micro_kernel!(Complex64, Cplx64Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Cplx64Vec, usize, usize) -> Cplx64Vec
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
        define_post_op_matmul_micro_kernel!(Complex64, Cplx64Vec, x8x1, 8, 1);
        define_neon_post_op_matmul_micro_kernel!(Complex64, Cplx64Vec, x8x2, 8, 2);
        [x8x1, x8x2][mr - 1]
    }

    fn get_horizontal_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Cplx64Vec, usize, usize) -> Cplx64Vec
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
        define_neon_post_op_matmul_micro_kernel!(Complex64, Cplx64Vec, x14x1, 14, 1);
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
avx2_kernels!(Complex64, Cplx64Vec);

#[cfg(target_feature = "avx512f")]
avx2_kernels!(Complex64, Cplx64Vec);