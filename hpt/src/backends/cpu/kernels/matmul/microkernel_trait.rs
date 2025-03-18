use crate::define_matmul_micro_kernel;
use hpt_common::Pointer;
use hpt_traits::tensor::CommonBounds;

/// register usage:
///
/// in `mma`: `C`: nr * mr, `A`: 1: `B`: nr
///
/// in `store`: 1 (reuse register used at `A` in mma)
///
/// `total register will use`: nr * (mr + 1).
///
/// Use this info to define microkernels, try not to exceed the register limit.
///
/// Try to let `nr * bits_register_can_hold = L1 cache line size`
///
/// # Example
///
/// ```rust
/// define_matmul_micro_kernel!(x2x6, 2, 6); // nr: 2, mr: 6
/// ```
pub trait MatmulMicroKernel
where
    Self: CommonBounds + Sized,
{
    fn get_kernel(
        nr: usize,
        mr: usize,
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        #[cfg(target_feature = "avx2")]
        {
            assert_eq!(nr, 2);
            // avx2 has 16 registers, each has 256 bits, assume cache line size is 512 bits
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
            assert_eq!(nr, 4);
            // sse has 16 registers, each has 128 bits, assume cache line size is 512 bits
            define_matmul_micro_kernel!(x4x1, 4, 1);
            define_matmul_micro_kernel!(x4x2, 4, 2);
            define_matmul_micro_kernel!(x4x3, 4, 3);
            return [x4x1, x4x2, x4x3][mr - 1];
        }
        #[cfg(target_feature = "neon")]
        {
            // neon has 32 registers, each has 128 bits, assume cache line size is 1024 bits
            unimplemented!()
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
        fn(*const MixedType::Vec) -> Self::Vec,
        fn(MixedType) -> Self,
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
            unimplemented!()
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
        #[cfg(target_feature = "avx2")]
        {
            2
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            4
        }
        #[cfg(target_feature = "neon")]
        {
            unimplemented!()
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
