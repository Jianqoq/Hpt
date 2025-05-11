use crate::microkernel_trait::MatmulMicroKernel;
use num_traits::ConstZero;
use crate::F32Vec;
use crate::Bf16Vec;
use std::ops::Add;

impl crate::Zero for half::bf16 {
    const ZERO: Self = half::bf16::from_f32_const(0.0);
}

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel<Bf16Vec, f32, F32Vec> for half::bf16 {
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
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x1, 2, 1);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x2, 2, 2);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x3, 2, 3);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x4, 2, 4);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x5, 2, 5);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x6, 2, 7);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x7, 2, 7);
        define_neon_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x8, 2, 8);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1]
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
        define_neon_matmul_micro_kernel!(half::bf16, Bf16Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Bf16Vec, usize, usize) -> Bf16Vec
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
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x1, 2, 1);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x2, 2, 2);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x3, 2, 3);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x4, 2, 4);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x5, 2, 5);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x6, 2, 7);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x7, 2, 7);
        define_neon_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x8, 2, 8);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1]
    }

    fn get_horizontal_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Bf16Vec, usize, usize) -> Bf16Vec
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
        define_neon_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x14x1, 14, 1);
        x14x1
    }

    fn get_max_mr() -> usize {
        8
    }

    fn get_max_nr() -> usize {
        2
    }

    fn get_horizontal_max_nr() -> usize {
        14
    }

    fn get_mixed_precision_kernel(
        nr: usize,
        mr: usize
    ) -> fn(
        crate::Pointer<f32>,
        crate::Pointer<f32>,
        crate::Pointer<half::bf16>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        fn(*mut Bf16Vec, *const F32Vec),
        fn(&mut half::bf16, &f32)
    ) {
        use crate::define_mixed_precision_matmul_micro_kernel;
        use crate::define_neon_mixed_precision_matmul_micro_kernel;
        assert_eq!(nr, 2);
        define_mixed_precision_matmul_micro_kernel!(half::bf16, Bf16Vec, f32, F32Vec, x2x1, 2, 1, 4);
        define_mixed_precision_matmul_micro_kernel!(half::bf16, Bf16Vec, f32, F32Vec, x2x2, 2, 2, 4);
        define_mixed_precision_matmul_micro_kernel!(half::bf16, Bf16Vec, f32, F32Vec, x2x3, 2, 3, 4);
        define_neon_mixed_precision_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x2x4,
            2,
            4,
            4
        );
        [x2x1, x2x2, x2x3, x2x4][mr - 1]
    }

    fn get_mixed_precision_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Bf16Vec, usize, usize) -> Bf16Vec
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
        fn(*mut Bf16Vec, *const F32Vec),
        fn(&mut Self, &f32),
        F,
        G
    ) {
        use crate::define_mixed_precision_post_op_matmul_micro_kernel;
        use crate::define_neon_mixed_precision_post_op_matmul_micro_kernel;
        assert_eq!(nr, 2);
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x2x1,
            2,
            1,
            4
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x2x2,
            2,
            2,
            4
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x2x3,
            2,
            3,
            4
        );
        define_neon_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x2x4,
            2,
            4,
            4
        );
        [x2x1, x2x2, x2x3, x2x4][mr - 1]
    }
}

#[cfg(target_feature = "avx2")]
impl MatmulMicroKernel<Bf16Vec, f32, F32Vec> for half::bf16 {
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
        assert_eq!(nr, 2);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x1, 2, 1);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x2, 2, 2);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x3, 2, 3);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x4, 2, 4);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x5, 2, 5);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x6, 2, 6);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1]
    }

    fn get_horizontal_kernel(
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
        assert_eq!(nr, 6);
        assert_eq!(mr, 1);
        define_matmul_micro_kernel!(half::bf16, Bf16Vec, x6x1, 6, 1);
        x6x1
    }

    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Bf16Vec, usize, usize) -> Bf16Vec,
    >(
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
        bool,
        usize,
        usize,
        F,
        G,
    ) {
        use crate::define_post_op_matmul_micro_kernel;
        assert_eq!(nr, 2);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x1, 2, 1);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x2, 2, 2);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x3, 2, 3);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x4, 2, 4);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x5, 2, 5);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x2x6, 2, 6);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1]
    }

    fn get_horizontal_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Bf16Vec, usize, usize) -> Bf16Vec,
    >(
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
        bool,
        usize,
        usize,
        F,
        G,
    ) {
        use crate::define_post_op_matmul_micro_kernel;
        assert_eq!(nr, 6);
        define_post_op_matmul_micro_kernel!(half::bf16, Bf16Vec, x6x1, 6, 1);
        x6x1
    }

    fn get_max_mr() -> usize {
        6
    }

    fn get_max_nr() -> usize {
        2
    }

    fn get_horizontal_max_nr() -> usize {
        6
    }

    fn get_mixed_precision_kernel(
        nr: usize,
        mr: usize,
    ) -> fn(
        crate::Pointer<f32>,
        crate::Pointer<f32>,
        crate::Pointer<half::bf16>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        fn(*mut Bf16Vec, *const F32Vec),
        fn(&mut half::bf16, &f32),
    ) {
        use crate::define_mixed_precision_matmul_micro_kernel;
        assert_eq!(nr, 1);
        define_mixed_precision_matmul_micro_kernel!(half::bf16, Bf16Vec, f32, F32Vec, x1x1, 1, 1, 2);
        define_mixed_precision_matmul_micro_kernel!(half::bf16, Bf16Vec, f32, F32Vec, x1x2, 1, 2, 2);
        define_mixed_precision_matmul_micro_kernel!(half::bf16, Bf16Vec, f32, F32Vec, x1x3, 1, 3, 2);
        define_mixed_precision_matmul_micro_kernel!(half::bf16, Bf16Vec, f32, F32Vec, x1x4, 1, 4, 2);
        define_mixed_precision_matmul_micro_kernel!(half::bf16, Bf16Vec, f32, F32Vec, x1x5, 1, 5, 2);
        define_mixed_precision_matmul_micro_kernel!(half::bf16, Bf16Vec, f32, F32Vec, x1x6, 1, 6, 2);
        [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6][mr - 1]
    }

    fn get_mixed_precision_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Bf16Vec, usize, usize) -> Bf16Vec,
    >(
        nr: usize,
        mr: usize,
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
        fn(*mut Bf16Vec, *const F32Vec),
        fn(&mut Self, &f32),
        F,
        G,
    ) {
        use crate::define_mixed_precision_post_op_matmul_micro_kernel;
        assert_eq!(nr, 1);
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x1x1,
            1,
            1,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x1x2,
            1,
            2,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x1x3,
            1,
            3,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x1x4,
            1,
            4,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x1x5,
            1,
            5,
            2
        );
        define_mixed_precision_post_op_matmul_micro_kernel!(
            half::bf16,
            Bf16Vec,
            f32,
            F32Vec,
            x1x6,
            1,
            6,
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
