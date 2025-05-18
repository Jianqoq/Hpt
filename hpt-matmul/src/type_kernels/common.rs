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
                assert_eq!(nr, 4);
                define_matmul_micro_kernel!($dtype, $vec_type, x4x1, 4, 1);
                define_matmul_micro_kernel!($dtype, $vec_type, x4x2, 4, 2);
                define_matmul_micro_kernel!($dtype, $vec_type, x4x3, 4, 3);
                define_matmul_micro_kernel!($dtype, $vec_type, x4x4, 4, 4);
                define_matmul_micro_kernel!($dtype, $vec_type, x4x5, 4, 5);
                define_matmul_micro_kernel!($dtype, $vec_type, x4x6, 4, 6);
                [x4x1, x4x2, x4x3, x4x4, x4x5, x4x6][mr - 1]
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
                assert_eq!(nr, 4);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x4x1, 4, 1);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x4x2, 4, 2);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x4x3, 4, 3);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x4x4, 4, 4);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x4x5, 4, 5);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x4x6, 4, 6);
                [x4x1, x4x2, x4x3, x4x4, x4x5, x4x6][mr - 1]
            }

            fn get_max_mr() -> usize {
                6
            }

            fn get_max_nr() -> usize {
                4
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
                use crate::microkernels::gemv_microkernel_impl;
                gemv_microkernel_impl!(8);
                gemv_microkernel::<$dtype, $vec_type>
            }

            fn get_gemv_nr() -> usize {
                8
            }
        }
    };
}

#[cfg(target_feature = "avx512f")]
pub(crate) use avx512_kernels;
