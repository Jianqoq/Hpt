use crate::microkernel_trait::MatmulMicroKernel;
use crate::I64Vec;
use num_traits::ConstZero;
use std::ops::Add;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
use crate::type_kernels::common::avx2_kernels;
#[cfg(target_feature = "avx512f")]
use crate::type_kernels::common::avx512_kernels;

#[cfg(target_feature = "neon")]
impl MatmulMicroKernel for i64 {
    fn get_kernel(
        nr: usize,
        mr: usize
    ) -> fn(crate::Pointer<Self>, crate::Pointer<Self>, crate::Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        use crate::define_matmul_micro_kernel;
        use crate::define_neon_matmul_micro_kernel;
        assert_eq!(nr, 8);
        define_matmul_micro_kernel!(i64, I64Vec, x8x1, 8, 1);
        define_neon_matmul_micro_kernel!(i64, I64Vec, x8x2, 8, 2);
        [x8x1, x8x2][mr - 1]
    }

    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(I64Vec, usize, usize) -> I64Vec
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
        assert_eq!(nr, 8);
        define_post_op_matmul_micro_kernel!(i64, I64Vec, x8x1, 8, 1);
        define_neon_post_op_matmul_micro_kernel!(i64, I64Vec, x8x2, 8, 2);
        [x8x1, x8x2][mr - 1]
    }

    fn get_max_mr() -> usize {
        2
    }

    fn get_max_nr() -> usize {
        8
    }
    
    type SelfVec = I64Vec;
    
    type MixedType = i64;
    
    type MixedVec = I64Vec;
    
    fn get_gemv_kernel() -> fn(
        a: crate::Pointer<Self>,
        b: crate::Pointer<Self>,
        c: crate::Pointer<Self>,
        n: usize,
        k: usize,
        ldb: i64,
        lhs_col_stride: i64
    ) {
        todo!()
    }
    
    fn get_gemv_nr() -> usize {
        todo!()
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
        todo!()
    }
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
avx2_kernels!(i64, I64Vec);

#[cfg(target_feature = "avx512f")]
avx512_kernels!(i64, I64Vec);