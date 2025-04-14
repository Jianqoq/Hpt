use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::TypeCommon;

use crate::backends::cpu::kernels::conv2d::microkernel_trait::Conv2dMicroKernel;

#[cfg(target_feature = "neon")]
impl Conv2dMicroKernel for crate::types::f16 {
    fn get_mixed_precision_kernel<MixedType>(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<Self>,
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<Self>,
        i64,
        i64,
        &mut i64,
        [i64; 3],
        [i64; 2],
        [i64; 2],
        [i64; 2],
        [i64; 2],
        [i64; 2],
        [i64; 2],
        [i64; 2],
        bool,
        fn(Self) -> MixedType,
        fn(MixedType) -> Self,
    )
    where
        MixedType: CommonBounds,
    {
        const {
            assert!(MixedType::BYTE_SIZE == 4);
        }
        use crate::conv2d_mixed_precision_micro_kernel;
        assert_eq!(nr, 2);
        // neon has 32 registers, each has 128 bits, assume cache line size is 1024 bits
        conv2d_mixed_precision_micro_kernel!(x2x1, 2, 1, 4);
        conv2d_mixed_precision_micro_kernel!(x2x2, 2, 2, 4);
        conv2d_mixed_precision_micro_kernel!(x2x3, 2, 3, 4);
        conv2d_mixed_precision_micro_kernel!(x2x4, 2, 4, 4);
        [x2x1, x2x2, x2x3, x2x4][mr - 1]
    }
    fn get_mixed_precision_kernel_with_post_op<
        MixedType,
        F: Fn(Self) -> Self,
        G: Fn(Self::Vec) -> Self::Vec,
    >(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        bool,
        fn(*const <MixedType as TypeCommon>::Vec) -> Self::Vec,
        fn(MixedType) -> Self,
        F,
        G,
    )
    where
        MixedType: CommonBounds,
    {
        unimplemented!()
    }
    fn get_max_mixed_precision_nr() -> usize {
        2
    }
    fn get_max_mixed_precision_mr() -> usize {
        4
    }
}

#[cfg(target_feature = "avx2")]
impl Conv2dMicroKernel for crate::types::f16 {
    fn get_mixed_precision_kernel<MixedType>(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<Self>,
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<Self>,
        i64,
        i64,
        &mut i64,
        [i64; 3],
        [i64; 2],
        [i64; 2],
        [i64; 2],
        [i64; 2],
        [i64; 2],
        [i64; 2],
        bool,
        fn(Self) -> MixedType,
        fn(MixedType) -> Self,
    )
    where
        MixedType: CommonBounds,
    {
        const {
            assert!(MixedType::BYTE_SIZE == 4);
        }
        use crate::conv2d_mixed_precision_micro_kernel;
        assert_eq!(nr, 1);
        // avx2 has 16 registers, each has 256 bits, assume cache line size is 512 bits
        conv2d_mixed_precision_micro_kernel!(x1x1, 1, 1, 2);
        conv2d_mixed_precision_micro_kernel!(x1x2, 1, 2, 2);
        conv2d_mixed_precision_micro_kernel!(x1x3, 1, 3, 2);
        conv2d_mixed_precision_micro_kernel!(x1x4, 1, 4, 2);
        conv2d_mixed_precision_micro_kernel!(x1x5, 1, 5, 2);
        conv2d_mixed_precision_micro_kernel!(x1x6, 1, 6, 2);
        [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6][mr - 1]
    }

    fn get_mixed_precision_kernel_with_post_op<
        MixedType,
        F: Fn(Self) -> Self,
        G: Fn(Self::Vec) -> Self::Vec,
    >(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        bool,
        fn(*const <MixedType as TypeCommon>::Vec) -> Self::Vec,
        fn(MixedType) -> Self,
        F,
        G,
    )
    where
        MixedType: CommonBounds,
    {
        const {
            assert!(MixedType::BYTE_SIZE == 4);
        }
        use crate::define_mixed_precision_post_op_matmul_micro_kernel;
        assert_eq!(nr, 1);
        // avx2 has 16 registers, each has 256 bits, assume cache line size is 512 bits
        define_mixed_precision_post_op_matmul_micro_kernel!(x1x1, 1, 1, 2);
        define_mixed_precision_post_op_matmul_micro_kernel!(x1x2, 1, 2, 2);
        define_mixed_precision_post_op_matmul_micro_kernel!(x1x3, 1, 3, 2);
        define_mixed_precision_post_op_matmul_micro_kernel!(x1x4, 1, 4, 2);
        define_mixed_precision_post_op_matmul_micro_kernel!(x1x5, 1, 5, 2);
        define_mixed_precision_post_op_matmul_micro_kernel!(x1x6, 1, 6, 2);
        [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6][mr - 1]
    }

    fn get_max_mixed_precision_nr() -> usize {
        1
    }
    fn get_max_mixed_precision_mr() -> usize {
        6
    }
}

#[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
impl MatmulMicroKernel for crate::types::f16 {
    fn get_mixed_precision_kernel<MixedType>(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        fn(*const <MixedType as TypeCommon>::Vec) -> Self::Vec,
        fn(MixedType) -> Self,
    )
    where
        MixedType: CommonBounds,
    {
        const {
            assert!(MixedType::BYTE_SIZE == 4);
        }
        use crate::define_mixed_precision_matmul_micro_kernel;
        assert_eq!(nr, 2);
        // sse has 16 registers, each has 128 bits, assume cache line size is 512 bits
        define_mixed_precision_matmul_micro_kernel!(x2x1, 2, 1, 4);
        define_mixed_precision_matmul_micro_kernel!(x2x2, 2, 2, 4);
        define_mixed_precision_matmul_micro_kernel!(x2x3, 2, 3, 4);
        [x2x1, x2x2, x2x3][mr - 1]
    }

    fn get_mixed_precision_kernel_with_post_op<
        MixedType,
        F: Fn(Self) -> Self,
        G: Fn(Self::Vec) -> Self::Vec,
    >(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        bool,
        fn(*const <MixedType as TypeCommon>::Vec) -> Self::Vec,
        fn(MixedType) -> Self,
        F,
        G,
    )
    where
        MixedType: CommonBounds,
    {
        const {
            assert!(MixedType::BYTE_SIZE == 4);
        }
        use crate::define_mixed_precision_post_op_matmul_micro_kernel;
        assert_eq!(nr, 2);
        // sse has 16 registers, each has 128 bits, assume cache line size is 512 bits
        define_mixed_precision_post_op_matmul_micro_kernel!(x2x1, 2, 1, 4);
        define_mixed_precision_post_op_matmul_micro_kernel!(x2x2, 2, 2, 4);
        define_mixed_precision_post_op_matmul_micro_kernel!(x2x3, 2, 3, 4);
        [x2x1, x2x2, x2x3][mr - 1]
    }

    fn get_max_mixed_precision_nr() -> usize {
        2
    }
    fn get_max_mixed_precision_mr() -> usize {
        3
    }
}
