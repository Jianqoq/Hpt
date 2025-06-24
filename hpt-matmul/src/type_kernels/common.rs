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
                assert_eq!(nr, 2);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x1, 2, 1);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x2, 2, 2);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x3, 2, 3);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x4, 2, 4);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x5, 2, 5);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x6, 2, 6);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x7, 2, 7);
                define_matmul_micro_kernel!($dtype, $vec_type, x2x8, 2, 8);
                [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1]
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
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x2x7, 2, 7);
                define_post_op_matmul_micro_kernel!($dtype, $vec_type, x2x8, 2, 8);
                [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8][mr - 1]
            }

            fn get_max_mr() -> usize {
                8
            }

            fn get_max_nr() -> usize {
                2
            }

            fn get_gemv_kernel() -> fn(
                a: crate::Pointer<Self>,
                b: crate::Pointer<Self>,
                c: crate::Pointer<Self>,
                n: usize,
                k: usize,
                ldb: i64,
                lhs_col_stride: i64,
            ) {
                use crate::microkernels::gemv_microkernel_impl;
                gemv_microkernel_impl!(8);
                gemv_microkernel::<$dtype, $vec_type>
            }

            fn get_gemv_kernel_with_post_op<
                F: Fn(Self, usize, usize) -> Self,
                F2: Fn(Self::SelfVec, usize, usize) -> Self::SelfVec,
            >() -> fn(
                a: crate::Pointer<Self>,
                b: crate::Pointer<Self>,
                c: crate::Pointer<Self>,
                n: usize,
                k: usize,
                ldb: i64,
                lhs_col_stride: i64,
                m_offset: usize,
                n_offset: usize,
                post_op: F,
                post_op_vec: F2,
            ) {
                use crate::microkernels::gemv_microkernel_post_op_impl;
                gemv_microkernel_post_op_impl!(8);
                gemv_microkernel_post_op::<$dtype, $vec_type, F, F2>
            }

            fn get_gemv_nr() -> usize {
                8
            }
        }
    };
}

#[cfg(target_feature = "avx512f")]
pub(crate) use avx512_kernels;
