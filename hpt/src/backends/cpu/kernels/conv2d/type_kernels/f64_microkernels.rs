use crate::backends::cpu::kernels::conv2d::microkernel_trait::Conv2dMicroKernel;

#[cfg(target_feature = "neon")]
impl Conv2dMicroKernel for f64 {
    fn get_kernel(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<Self>,
        hpt_common::Pointer<Self>,
        hpt_common::Pointer<Self>,
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
        use crate::conv2d_micro_kernel;
        use crate::conv2d_neon_micro_kernel;
        assert_eq!(nr, 8);
        conv2d_micro_kernel!(x8x1, 8, 1);
        conv2d_neon_micro_kernel!(x8x2, 8, 2);
        [x8x1, x8x2][mr - 1]
    }
    fn get_kernel_with_post_op<F: Fn(Self) -> Self, G: Fn(Self::Vec) -> Self::Vec>(
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
        bool,
        F,
        G,
    ) {
        unimplemented!()
    }
    fn get_max_mr() -> usize {
        2
    }
    fn get_max_nr() -> usize {
        8
    }
}

#[cfg(not(target_feature = "neon"))]
impl Conv2dMicroKernel for f64 {}
