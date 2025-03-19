use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::TypeCommon;

use super::microkernel_trait::MatmulMicroKernel;

impl MatmulMicroKernel for crate::types::f16 {
    #[cfg(target_feature = "neon")]
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
        use crate::define_matmul_micro_kernel;
        assert_eq!(nr, 2);
        define_matmul_micro_kernel!(x2x1, 2, 1);
        define_matmul_micro_kernel!(x2x2, 2, 2);
        define_matmul_micro_kernel!(x2x3, 2, 3);
        define_matmul_micro_kernel!(x2x4, 2, 4);
        define_matmul_micro_kernel!(x2x5, 2, 5);
        define_matmul_micro_kernel!(x2x6, 2, 7);
        define_matmul_micro_kernel!(x2x7, 2, 7);
        define_neon_matmul_micro_kernel!(x2x8, 2, 8);
        [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1]
    }
    #[cfg(target_feature = "neon")]
    fn get_max_mr() -> usize {
        8
    }
    #[cfg(target_feature = "neon")]
    fn get_max_nr() -> usize {
        2
    }
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
        assert_eq!(MixedType::BYTE_SIZE, 4);
        #[cfg(target_feature = "avx2")]
        {
            use crate::define_mixed_precision_matmul_micro_kernel;
            assert_eq!(nr, 1);
            // avx2 has 16 registers, each has 256 bits, assume cache line size is 512 bits
            define_mixed_precision_matmul_micro_kernel!(x1x1, 1, 1, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x2, 1, 2, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x3, 1, 3, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x4, 1, 4, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x5, 1, 5, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x6, 1, 6, 2);
            return [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6][mr - 1];
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            use crate::define_mixed_precision_matmul_micro_kernel;
            assert_eq!(nr, 2);
            // sse has 16 registers, each has 128 bits, assume cache line size is 512 bits
            define_mixed_precision_matmul_micro_kernel!(x2x1, 2, 1, 4);
            define_mixed_precision_matmul_micro_kernel!(x2x2, 2, 2, 4);
            define_mixed_precision_matmul_micro_kernel!(x2x3, 2, 3, 4);
            return [x2x1, x2x2, x2x3][mr - 1];
        }
        #[cfg(target_feature = "neon")]
        {
            use crate::define_mixed_precision_matmul_micro_kernel;
            assert_eq!(nr, 2);
            // neon has 32 registers, each has 128 bits, assume cache line size is 1024 bits
            define_mixed_precision_matmul_micro_kernel!(x2x1, 2, 1, 4);
            define_mixed_precision_matmul_micro_kernel!(x2x2, 2, 2, 4);
            define_mixed_precision_matmul_micro_kernel!(x2x3, 2, 3, 4);
            define_neon_mixed_precision_matmul_micro_kernel!(x2x4, 2, 4, 4);
            return [x2x1, x2x2, x2x3, x2x4][mr - 1];
        }
        #[cfg(all(
            not(target_feature = "avx2"),
            not(target_feature = "sse"),
            not(target_feature = "neon")
        ))]
        {
            unimplemented!()
        }
    }

    fn get_max_mixed_precision_nr() -> usize {
        #[cfg(target_feature = "avx2")]
        {
            1
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            2
        }
        #[cfg(target_feature = "neon")]
        {
            2
        }
        #[cfg(all(
            not(target_feature = "avx2"),
            not(target_feature = "sse"),
            not(target_feature = "neon")
        ))]
        {
            unimplemented!()
        }
    }
    fn get_max_mixed_precision_mr() -> usize {
        #[cfg(target_feature = "avx2")]
        {
            6
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            3
        }
        #[cfg(target_feature = "neon")]
        {
            4
        }
        #[cfg(all(
            not(target_feature = "avx2"),
            not(target_feature = "sse"),
            not(target_feature = "neon")
        ))]
        {
            unimplemented!()
        }
    }
}
