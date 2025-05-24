use matconv_simd::VecTrait;

use crate::Pointer;

pub trait Conv2dMicroKernel
where
    Self: Sized + Copy,
{
    #[allow(private_bounds)]
    type SelfVec: VecTrait<Self> + std::ops::Mul<Output = Self::SelfVec> + Copy;
    type MixedType;
    type MixedVec;
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
        [i64; 2],
        bool,
    );
    // {
    //     #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    //     {
    //         use crate::conv2d_micro_kernel;
    //         assert_eq!(nr, 2);
    //         conv2d_micro_kernel!(x2x1, 2, 1);
    //         conv2d_micro_kernel!(x2x2, 2, 2);
    //         conv2d_micro_kernel!(x2x3, 2, 3);
    //         conv2d_micro_kernel!(x2x4, 2, 4);
    //         conv2d_micro_kernel!(x2x5, 2, 5);
    //         return [x2x1, x2x2, x2x3, x2x4, x2x5][mr - 1];
    //     }
    //     #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
    //     {
    //         use crate::conv2d_micro_kernel;
    //         assert_eq!(nr, 4);
    //         conv2d_micro_kernel!(x4x1, 4, 1);
    //         conv2d_micro_kernel!(x4x2, 4, 2);
    //         conv2d_micro_kernel!(x4x3, 4, 3);
    //         return [x4x1, x4x2, x4x3][mr - 1];
    //     }
    //     #[cfg(target_feature = "neon")]
    //     {
    //         use crate::conv2d_micro_kernel;
    //         assert_eq!(nr, 4);
    //         conv2d_micro_kernel!(x4x1, 4, 1);
    //         conv2d_micro_kernel!(x4x2, 4, 2);
    //         conv2d_micro_kernel!(x4x3, 4, 3);
    //         conv2d_micro_kernel!(x4x4, 4, 4);
    //         conv2d_micro_kernel!(x4x5, 4, 5);
    //         conv2d_micro_kernel!(x4x6, 4, 6);
    //         return [x4x1, x4x2, x4x3, x4x4, x4x5, x4x6][mr - 1];
    //     }
    // }

    #[allow(unused_variables)]
    fn get_kernel_with_padding(
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
        [i64; 2],
        bool,
    ); 
    // {
    //     #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    //     {
    //         use crate::conv2d_micro_kernel_with_padding;
    //         assert_eq!(nr, 2);
    //         conv2d_micro_kernel_with_padding!(x2x1, 2, 1);
    //         conv2d_micro_kernel_with_padding!(x2x2, 2, 2);
    //         conv2d_micro_kernel_with_padding!(x2x3, 2, 3);
    //         conv2d_micro_kernel_with_padding!(x2x4, 2, 4);
    //         conv2d_micro_kernel_with_padding!(x2x5, 2, 5);
    //         conv2d_micro_kernel_with_padding!(x2x6, 2, 6);
    //         return [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1];
    //     }
    //     #[cfg(target_feature = "avx512f")]
    //     {
    //         use crate::conv2d_micro_kernel_with_padding;
    //         assert_eq!(nr, 2);
    //         conv2d_micro_kernel_with_padding!(x2x1, 2, 1);
    //         conv2d_micro_kernel_with_padding!(x2x2, 2, 2);
    //         conv2d_micro_kernel_with_padding!(x2x3, 2, 3);
    //         conv2d_micro_kernel_with_padding!(x2x4, 2, 4);
    //         conv2d_micro_kernel_with_padding!(x2x5, 2, 5);
    //         conv2d_micro_kernel_with_padding!(x2x6, 2, 6);
    //         conv2d_micro_kernel_with_padding!(x2x7, 2, 7);
    //         conv2d_micro_kernel_with_padding!(x2x8, 2, 8);
    //         return [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1];
    //     }
    //     #[cfg(target_feature = "neon")]
    //     {
    //         use crate::conv2d_micro_kernel_with_padding;
    //         assert_eq!(nr, 4);
    //         conv2d_micro_kernel_with_padding!(x4x1, 4, 1);
    //         conv2d_micro_kernel_with_padding!(x4x2, 4, 2);
    //         conv2d_micro_kernel_with_padding!(x4x3, 4, 3);
    //         conv2d_micro_kernel_with_padding!(x4x4, 4, 4);
    //         conv2d_micro_kernel_with_padding!(x4x5, 4, 5);
    //         conv2d_micro_kernel_with_padding!(x4x6, 4, 6);
    //         return [x4x1, x4x2, x4x3, x4x4, x4x5, x4x6][mr - 1];
    //     }
    //     #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
    //     {
    //         use crate::conv2d_micro_kernel_with_padding;
    //         assert_eq!(nr, 4);
    //         conv2d_micro_kernel_with_padding!(x4x1, 4, 1);
    //         conv2d_micro_kernel_with_padding!(x4x2, 4, 2);
    //         conv2d_micro_kernel_with_padding!(x4x3, 4, 3);
    //         return [x4x1, x4x2, x4x3][mr - 1];
    //     }
    // }

    #[allow(unused_variables)]
    fn get_mixed_precision_kernel(
        nr: usize,
        mr: usize,
    ) -> fn(
        crate::Pointer<Self::MixedType>,
        crate::Pointer<Self::MixedType>,
        crate::Pointer<Self>,
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
        fn(*mut Self::MixedVec, *const Self),
        fn(*mut Self::SelfVec, *const Self::MixedVec),
        fn(Self) -> Self::MixedType,
        fn(&mut Self, &Self::MixedType),
    )
    {
        unimplemented!("mixed precision kernel is required for user to implement")
    }

    #[allow(unused_variables)]
    fn get_mixed_precision_kernel_with_padding(
        nr: usize,
        mr: usize,
    ) -> fn(
        crate::Pointer<Self::MixedType>,
        crate::Pointer<Self::MixedType>,
        crate::Pointer<Self>,
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
        fn(*mut Self::MixedVec, *const Self),
        fn(*mut Self::SelfVec, *const Self::MixedVec),
        fn(Self) -> Self::MixedType,
        fn(&mut Self, &Self::MixedType),
    )
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
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            5
        }
        #[cfg(target_feature = "avx512f")]
        {
            8
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            3
        }
        #[cfg(target_feature = "neon")]
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
    fn get_max_nr() -> usize {
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            2
        }
        #[cfg(target_feature = "avx512f")]
        {
            2
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            4
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
