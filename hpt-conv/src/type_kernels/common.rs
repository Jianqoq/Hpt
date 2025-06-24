#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
macro_rules! avx2_kernels {
    ($dtype:ty, $vec_type:ty) => {
        impl MatmulMicroKernel for $dtype {
            type SelfVec = $vec_type;
            type MixedType = $dtype;
            type MixedVec = $vec_type;
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
                define_matmul_micro_kernel!($dtype, $vec_type, x2x1, 2, 1);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x2, 2, 2);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x3, 2, 3);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x4, 2, 4);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x5, 2, 5);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x6, 2, 6);
                [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1]
            }

            fn get_kernel_with_post_op<
                F: Fn(Self, usize, usize) -> Self,
                G: Fn($vec_type, usize, usize) -> $vec_type,
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
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x2x1, 2, 1);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x2x2, 2, 2);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x2x3, 2, 3);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x2x4, 2, 4);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x2x5, 2, 5);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x2x6, 2, 6);
                [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6][mr - 1]
            }

            fn get_max_mr() -> usize {
                6
            }

            fn get_max_nr() -> usize {
                2
            }
        }
    };
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub(crate) use avx2_kernels;

#[cfg(target_feature = "avx512f")]
macro_rules! avx512_kernels {
    ($dtype:ty, $vec_type:ty) => {
        impl Conv2dMicroKernel for $dtype {
            type SelfVec = $vec_type;
            type MixedType = $dtype;
            type MixedVec = $vec_type;
            #[allow(unused_variables)]
            fn get_kernel(
                nr: usize,
                mr: usize,
            ) -> fn(
                crate::Pointer<Self>,
                crate::Pointer<Self>,
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
            ) {
                use crate::conv2d_micro_kernel;
                assert_eq!(nr, 2);
                conv2d_micro_kernel!(x2x1, $dtype, $vec_type, 2, 1);
                conv2d_micro_kernel!(x2x2, $dtype, $vec_type, 2, 2);
                conv2d_micro_kernel!(x2x3, $dtype, $vec_type, 2, 3);
                conv2d_micro_kernel!(x2x4, $dtype, $vec_type, 2, 4);
                conv2d_micro_kernel!(x2x5, $dtype, $vec_type, 2, 5);
                conv2d_micro_kernel!(x2x6, $dtype, $vec_type, 2, 6);
                conv2d_micro_kernel!(x2x7, $dtype, $vec_type, 2, 7);
                conv2d_micro_kernel!(x2x8, $dtype, $vec_type, 2, 8);
                return [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1];
            }

            #[allow(unused_variables)]
            fn get_kernel_with_padding(
                nr: usize,
                mr: usize,
            ) -> fn(
                crate::Pointer<Self>,
                crate::Pointer<Self>,
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
            ) {
                use crate::conv2d_micro_kernel_with_padding;
                assert_eq!(nr, 2);
                conv2d_micro_kernel_with_padding!(x2x1, $dtype, $vec_type, 2, 1);
                conv2d_micro_kernel_with_padding!(x2x2, $dtype, $vec_type, 2, 2);
                conv2d_micro_kernel_with_padding!(x2x3, $dtype, $vec_type, 2, 3);
                conv2d_micro_kernel_with_padding!(x2x4, $dtype, $vec_type, 2, 4);
                conv2d_micro_kernel_with_padding!(x2x5, $dtype, $vec_type, 2, 5);
                conv2d_micro_kernel_with_padding!(x2x6, $dtype, $vec_type, 2, 6);
                conv2d_micro_kernel_with_padding!(x2x7, $dtype, $vec_type, 2, 7);
                conv2d_micro_kernel_with_padding!(x2x8, $dtype, $vec_type, 2, 8);
                return [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1];
            }

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
            ) {
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
            ) {
                unimplemented!("mixed precision kernel is required for user to implement")
            }

            fn get_max_mixed_precision_mr() -> usize {
                unimplemented!()
            }
            fn get_max_mixed_precision_nr() -> usize {
                unimplemented!()
            }

            fn get_max_mr() -> usize {
                8
            }
            fn get_max_nr() -> usize {
                2
            }
        }
    };
}

#[cfg(target_feature = "avx512f")]
pub(crate) use avx512_kernels;
