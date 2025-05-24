use crate::microkernel_trait::MatmulMicroKernel;
use crate::F32Vec;
use num_traits::ConstZero;
use std::ops::Add;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
use crate::type_kernels::common::avx2_kernels;

#[cfg(target_feature = "avx512f")]
use crate::type_kernels::common::avx512_kernels;

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for f32 {
    #[allow(unused_variables)]
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
        define_matmul_micro_kernel!(f32, F32Vec, x4x1, 4, 1);
        define_matmul_micro_kernel!(f32, F32Vec, x4x2, 4, 2);
        define_matmul_micro_kernel!(f32, F32Vec, x4x3, 4, 3);
        define_neon_matmul_micro_kernel!(f32, F32Vec, x4x4, 4, 4);
        [x4x1, x4x2, x4x3, x4x4][mr - 1]
    }

    #[allow(unused_variables)]
    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(F32Vec, usize, usize) -> F32Vec
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
        define_post_op_matmul_micro_kernel!(f32, F32Vec, x4x1, 4, 1);
        define_post_op_matmul_micro_kernel!(f32, F32Vec, x4x2, 4, 2);
        define_post_op_matmul_micro_kernel!(f32, F32Vec, x4x3, 4, 3);
        define_neon_post_op_matmul_micro_kernel!(f32, F32Vec, x4x4, 4, 4);
        [x4x1, x4x2, x4x3, x4x4][mr - 1]
    }

    fn get_max_mr() -> usize {
        4
    }

    fn get_max_nr() -> usize {
        4
    }

    type SelfVec = F32Vec;

    type MixedType = f32;

    type MixedVec = F32Vec;

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
        gemv_microkernel::<f32, F32Vec>
    }

    fn get_gemv_nr() -> usize {
        8
    }
    
    fn get_gemv_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        F2: Fn(Self::SelfVec, usize, usize) -> Self::SelfVec
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
        post_op_vec: F2
    ) {
        use crate::microkernels::gemv_microkernel_post_op_impl;
        gemv_microkernel_post_op_impl!(8);
        gemv_microkernel_post_op::<f32, F32Vec, F, F2>
    }
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
avx2_kernels!(f32, F32Vec);

#[cfg(target_feature = "avx512f")]
avx512_kernels!(f32, F32Vec);
