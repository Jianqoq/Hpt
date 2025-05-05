use hpt_common::Pointer;
use hpt_traits::tensor::CommonBounds;

/// A trait for microkernels of matrix multiplication
pub trait MatmulMicroKernel
where
    Self: CommonBounds + Sized,
{
    #[allow(unused_variables)]
    fn get_kernel(
        nr: usize,
        mr: usize,
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        #[cfg(target_feature = "avx2")]
        {
            use crate::define_matmul_micro_kernel;
            assert_eq!(nr, 2);
            define_matmul_micro_kernel!(x2x1, 2, 1);
            define_matmul_micro_kernel!(x2x2, 2, 2);
            define_matmul_micro_kernel!(x2x3, 2, 3);
            define_matmul_micro_kernel!(x2x4, 2, 4);
            define_matmul_micro_kernel!(x2x5, 2, 5);
            define_matmul_micro_kernel!(x2x6, 2, 6);
            return [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1];
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            use crate::define_matmul_micro_kernel;
            assert_eq!(nr, 2);
            define_matmul_micro_kernel!(x2x1, 2, 1);
            define_matmul_micro_kernel!(x2x2, 2, 2);
            define_matmul_micro_kernel!(x2x3, 2, 3);
            return [x2x1, x2x2, x2x3][mr - 1];
        }
        #[cfg(target_feature = "neon")]
        {
            use crate::define_matmul_micro_kernel;
            assert_eq!(nr, 8);
            define_matmul_micro_kernel!(x8x1, 8, 1);
            define_matmul_micro_kernel!(x8x2, 8, 2);
            return [x8x1, x8x2][mr - 1];
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

    #[allow(unused_variables)]
    fn get_horizontal_kernel(
        nr: usize,
        mr: usize,
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        #[cfg(target_feature = "avx2")]
        {
            use crate::define_matmul_micro_kernel;
            assert_eq!(nr, 6);
            assert_eq!(mr, 1);
            define_matmul_micro_kernel!(x6x1, 6, 1);
            return x6x1;
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

    #[allow(unused_variables)]
    fn get_inline_asm_kernel(
        nr: usize,
        mr: usize,
        has_rem: bool,
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        unimplemented!("inline asm kernel only support specific microkernel")
    }

    #[allow(unused_variables)]
    fn get_kernel_with_post_op<F: Fn(Self, usize, usize) -> Self, G: Fn(Self::Vec, usize, usize) -> Self::Vec>(
        nr: usize,
        mr: usize,
    ) -> fn(
        Pointer<Self>,
        Pointer<Self>,
        Pointer<Self>,
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
        #[cfg(target_feature = "avx2")]
        {
            use crate::define_post_op_matmul_micro_kernel;
            assert_eq!(nr, 2);
            define_post_op_matmul_micro_kernel!(x2x1, 2, 1);
            define_post_op_matmul_micro_kernel!(x2x2, 2, 2);
            define_post_op_matmul_micro_kernel!(x2x3, 2, 3);
            define_post_op_matmul_micro_kernel!(x2x4, 2, 4);
            define_post_op_matmul_micro_kernel!(x2x5, 2, 5);
            define_post_op_matmul_micro_kernel!(x2x6, 2, 6);
            return [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1];
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            use crate::define_post_op_matmul_micro_kernel;
            assert_eq!(nr, 4);
            define_post_op_matmul_micro_kernel!(x4x1, 4, 1);
            define_post_op_matmul_micro_kernel!(x4x2, 4, 2);
            define_post_op_matmul_micro_kernel!(x4x3, 4, 3);
            return [x4x1, x4x2, x4x3][mr - 1];
        }
        #[cfg(target_feature = "neon")]
        {
            use crate::define_post_op_matmul_micro_kernel;
            assert_eq!(nr, 8);
            define_post_op_matmul_micro_kernel!(x8x1, 8, 1);
            define_post_op_matmul_micro_kernel!(x8x2, 8, 2);
            return [x8x1, x8x2][mr - 1];
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

    #[allow(unused_variables)]
    fn get_mixed_precision_kernel<MixedType>(
        nr: usize,
        mr: usize,
    ) -> fn(
        Pointer<MixedType>,
        Pointer<MixedType>,
        Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        fn(*mut Self::Vec, *const MixedType::Vec),
        fn(&mut Self, &MixedType),
    )
    where
        MixedType: CommonBounds,
    {
        unimplemented!("mixed precision kernel is required for user to implement")
    }

    #[allow(unused_variables)]
    fn get_mixed_precision_kernel_with_post_op<
        MixedType,
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Self::Vec, usize, usize) -> Self::Vec,
    >(
        nr: usize,
        mr: usize,
    ) -> fn(
        Pointer<MixedType>,
        Pointer<MixedType>,
        Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        bool,
        usize,
        usize,
        fn(*mut Self::Vec, *const MixedType::Vec),
        fn(&mut Self, &MixedType),
        F,
        G,
    )
    where
        MixedType: CommonBounds,
    {
        unimplemented!("mixed precision kernel is required for user to implement")
    }

    fn get_max_mixed_precision_mr() -> usize {
        unimplemented!()
    }
    fn get_max_mixed_precision_nr() -> usize {
        unimplemented!()
    }

    fn get_max_mr() -> usize {
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
    fn get_max_nr() -> usize {
        #[cfg(any(target_feature = "avx2", target_feature = "sse"))]
        {
            2
        }
        #[cfg(target_feature = "neon")]
        {
            8
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
    fn get_horizontal_max_nr() -> usize {
        #[cfg(any(target_feature = "avx2", target_feature = "sse"))]
        {
            6
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
