use hpt_common::Pointer;
use hpt_traits::tensor::CommonBounds;

pub trait Conv2dMicroKernel
where
    Self: CommonBounds + Sized,
{
    #[allow(unused_variables)]
    fn get_kernel(
        nr: usize,
        mr: usize,
    ) -> fn(
        Pointer<Self>,
        Pointer<Self>,
        Pointer<Self>,
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
    ) {
        // #[cfg(target_feature = "avx2")]
        {
            use crate::conv2d_micro_kernel;
            assert_eq!(nr, 2);
            // avx2 has 16 registers, each has 256 bits, assume cache line size is 512 bits
            conv2d_micro_kernel!(x2x1, 2, 1);
            conv2d_micro_kernel!(x2x2, 2, 2);
            conv2d_micro_kernel!(x2x3, 2, 3);
            conv2d_micro_kernel!(x2x4, 2, 4);
            conv2d_micro_kernel!(x2x5, 2, 5);
            conv2d_micro_kernel!(x2x6, 2, 6);
            return [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1];
        }
        // #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        // {
        //     use crate::conv2d_micro_kernel;
        //     assert_eq!(nr, 4);
        //     // sse has 16 registers, each has 128 bits, assume cache line size is 512 bits
        //     conv2d_micro_kernel!(x4x1, 4, 1);
        //     conv2d_micro_kernel!(x4x2, 4, 2);
        //     conv2d_micro_kernel!(x4x3, 4, 3);
        //     return [x4x1, x4x2, x4x3][mr - 1];
        // }
        // #[cfg(target_feature = "neon")]
        // {
        //     use crate::conv2d_micro_kernel;
        //     assert_eq!(nr, 8);
        //     conv2d_micro_kernel!(x8x1, 8, 1);
        //     conv2d_micro_kernel!(x8x2, 8, 2);
        //     return [x8x1, x8x2][mr - 1];
        // }
        // #[cfg(all(
        //     not(target_feature = "avx2"),
        //     not(target_feature = "sse"),
        //     not(target_feature = "neon")
        // ))]
        // {
        //     unimplemented!()
        // }
    }

    #[allow(unused_variables)]
    fn get_kernel_with_post_op<F: Fn(Self) -> Self, G: Fn(Self::Vec) -> Self::Vec>(
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
        F,
        G,
    ) {
        unimplemented!()
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

    #[allow(unused_variables)]
    fn get_mixed_precision_kernel_with_post_op<
        MixedType,
        F: Fn(Self) -> Self,
        G: Fn(Self::Vec) -> Self::Vec,
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
        fn(*const MixedType::Vec) -> Self::Vec,
        fn(MixedType) -> Self,
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
}
