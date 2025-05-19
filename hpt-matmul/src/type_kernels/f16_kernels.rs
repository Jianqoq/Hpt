use crate::F16Vec;
use crate::F32Vec;
use crate::microkernel_trait::MatmulMicroKernel;
use num_traits::ConstZero;
use std::ops::Add;

impl crate::Zero for half::f16 {
    const ZERO: Self = half::f16::from_f32_const(0.0);
}

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for half::f16 {
    type SelfVec = F16Vec;
    type MixedType = f32;
    type MixedVec = F32Vec;

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
        assert_eq!(nr, 2);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x1, 2, 1);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x2, 2, 2);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x3, 2, 3);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x4, 2, 4);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x5, 2, 5);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x6, 2, 7);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x7, 2, 7);
        define_neon_matmul_micro_kernel!(half::f16, F16Vec, x2x8, 2, 8);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1]
    }

    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(F16Vec, usize, usize) -> F16Vec
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
        assert_eq!(nr, 2);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x1, 2, 1);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x2, 2, 2);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x3, 2, 3);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x4, 2, 4);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x5, 2, 5);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x6, 2, 7);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x7, 2, 7);
        define_neon_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x8, 2, 8);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1]
    }

    fn get_max_mr() -> usize {
        8
    }
    fn get_max_mixed_precision_mr() -> usize {
        4
    }
    fn get_max_nr() -> usize {
        2
    }
    fn get_max_mixed_precision_nr() -> usize {
        2
    }

    fn get_mixed_precision_kernel(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<f32>,
        crate::Pointer<f32>,
        crate::Pointer<half::f16>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        fn(*mut F16Vec, *const F32Vec),
        fn(&mut half::f16, &f32)
    ) {
        use crate::define_mixed_precision_matmul_micro_kernel;
        use crate::define_neon_mixed_precision_matmul_micro_kernel;
        assert_eq!(nr, 2);
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x2x1,
            2,
            1,
            4,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x2x2,
            2,
            2,
            4,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x2x3,
            2,
            3,
            4,
            2
        );
        define_neon_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x2x4,
            2,
            4,
            4,
            2
        );
        [x2x1, x2x2, x2x3, x2x4][mr - 1]
    }

    fn get_mixed_precision_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(F16Vec, usize, usize) -> F16Vec
    >(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<f32>,
        crate::Pointer<f32>,
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
        fn(*mut F16Vec, *const F32Vec),
        fn(&mut Self, &f32),
        F,
        G
    ) {
        use crate::define_mixed_precision_post_op_matmul_micro_kernel;
        use crate::define_neon_mixed_precision_post_op_matmul_micro_kernel;
        assert_eq!(nr, 2);
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x2x1,
            2,
            1,
            4,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x2x2,
            2,
            2,
            4,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x2x3,
            2,
            3,
            4,
            2
        );
        define_neon_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x2x4,
            2,
            4,
            4,
            2
        );
        [x2x1, x2x2, x2x3, x2x4][mr - 1]
    }

    fn get_gemv_kernel() -> fn(
        a: crate::Pointer<Self>,
        b: crate::Pointer<Self>,
        c: crate::Pointer<Self>,
        n: usize,
        k: usize,
        ldb: i64,
        lhs_col_stride: i64
    ) {
        todo!()
    }

    fn get_gemv_nr() -> usize {
        todo!()
    }
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
impl MatmulMicroKernel for half::f16 {
    type SelfVec = F16Vec;
    type MixedType = f32;
    type MixedVec = F32Vec;
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
        assert_eq!(nr, 2);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x1, 2, 1);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x2, 2, 2);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x3, 2, 3);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x4, 2, 4);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x5, 2, 5);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x6, 2, 6);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1]
    }

    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(F16Vec, usize, usize) -> F16Vec
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
        assert_eq!(nr, 2);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x1, 2, 1);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x2, 2, 2);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x3, 2, 3);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x4, 2, 4);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x5, 2, 5);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x6, 2, 6);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1]
    }

    fn get_max_mr() -> usize {
        6
    }

    fn get_max_nr() -> usize {
        2
    }

    fn get_mixed_precision_kernel(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<f32>,
        crate::Pointer<f32>,
        crate::Pointer<half::f16>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        fn(*mut F16Vec, *const F32Vec),
        fn(&mut half::f16, &f32)
    ) {
        use crate::define_mixed_precision_matmul_micro_kernel;
        assert_eq!(nr, 1);
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x1,
            1,
            1,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x2,
            1,
            2,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x3,
            1,
            3,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x4,
            1,
            4,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x5,
            1,
            5,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x6,
            1,
            6,
            2,
            2
        );
        [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6][mr - 1]
    }

    fn get_mixed_precision_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(F16Vec, usize, usize) -> F16Vec
    >(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<f32>,
        crate::Pointer<f32>,
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
        fn(*mut F16Vec, *const F32Vec),
        fn(&mut Self, &f32),
        F,
        G
    ) {
        use crate::define_mixed_precision_post_op_matmul_micro_kernel;
        assert_eq!(nr, 1);
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x1,
            1,
            1,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x2,
            1,
            2,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x3,
            1,
            3,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x4,
            1,
            4,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x5,
            1,
            5,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x6,
            1,
            6,
            2,
            2
        );
        [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6][mr - 1]
    }
    fn get_max_mixed_precision_nr() -> usize {
        1
    }
    fn get_max_mixed_precision_mr() -> usize {
        6
    }
}

#[cfg(target_feature = "avx512f")]
impl MatmulMicroKernel for half::f16 {
    type SelfVec = F16Vec;
    type MixedType = f32;
    type MixedVec = F32Vec;
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
        assert_eq!(nr, 2);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x1, 2, 1);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x2, 2, 2);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x3, 2, 3);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x4, 2, 4);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x5, 2, 5);
        define_matmul_micro_kernel!(half::f16, F16Vec, x2x6, 2, 6);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1]
    }

    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(F16Vec, usize, usize) -> F16Vec
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
        assert_eq!(nr, 2);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x1, 2, 1);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x2, 2, 2);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x3, 2, 3);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x4, 2, 4);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x5, 2, 5);
        define_post_op_matmul_micro_kernel!(half::f16, F16Vec, x2x6, 2, 6);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1]
    }

    fn get_max_mr() -> usize {
        6
    }

    fn get_max_nr() -> usize {
        2
    }

    fn get_mixed_precision_kernel(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<f32>,
        crate::Pointer<f32>,
        crate::Pointer<half::f16>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        fn(*mut F16Vec, *const F32Vec),
        fn(&mut half::f16, &f32)
    ) {
        use crate::define_mixed_precision_matmul_micro_kernel;
        assert_eq!(nr, 1);
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x1,
            1,
            1,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x2,
            1,
            2,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x3,
            1,
            3,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x4,
            1,
            4,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x5,
            1,
            5,
            2,
            2
        );
        define_mixed_precision_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x6,
            1,
            6,
            2,
            2
        );
        [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6][mr - 1]
    }

    fn get_mixed_precision_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(F16Vec, usize, usize) -> F16Vec
    >(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<f32>,
        crate::Pointer<f32>,
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
        fn(*mut F16Vec, *const F32Vec),
        fn(&mut Self, &f32),
        F,
        G
    ) {
        use crate::define_mixed_precision_post_op_matmul_micro_kernel;
        assert_eq!(nr, 1);
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x1,
            1,
            1,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x2,
            1,
            2,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x3,
            1,
            3,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x4,
            1,
            4,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x5,
            1,
            5,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::f16,
            F16Vec,
            f32,
            F32Vec,
            x1x6,
            1,
            6,
            2,
            2
        );
        [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6][mr - 1]
    }
    fn get_max_mixed_precision_nr() -> usize {
        1
    }
    fn get_max_mixed_precision_mr() -> usize {
        6
    }

    fn get_gemv_kernel() -> fn(
        a: crate::Pointer<Self>,
        b: crate::Pointer<Self>,
        c: crate::Pointer<Self>,
        n: usize,
        k: usize,
        ldb: i64,
        lhs_col_stride: i64
    ) {
        todo!()
    }

    fn get_gemv_nr() -> usize {
        todo!()
    }
}
