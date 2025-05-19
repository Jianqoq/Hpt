use crate::{ Pointer, VecTrait };

/// A trait for microkernels of matrix multiplication
pub trait MatmulMicroKernel where Self: Sized + Copy + crate::Zero {
    #[allow(private_bounds)]
    type SelfVec: VecTrait<Self> + std::ops::Mul<Output = Self::SelfVec> + Copy;
    type MixedType;
    type MixedVec;

    fn get_kernel(
        nr: usize,
        mr: usize
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool);
    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Self::SelfVec, usize, usize) -> Self::SelfVec
    >(
        nr: usize,
        mr: usize
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
        G
    );
    #[allow(unused_variables)]
    fn get_mixed_precision_kernel(
        nr: usize,
        mr: usize
    ) -> fn(
        Pointer<Self::MixedType>,
        Pointer<Self::MixedType>,
        Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        fn(*mut Self::SelfVec, *const Self::MixedVec),
        fn(&mut Self, &Self::MixedType)
    ) {
        unimplemented!()
    }
    #[allow(unused_variables)]
    fn get_mixed_precision_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Self::SelfVec, usize, usize) -> Self::SelfVec
    >(
        nr: usize,
        mr: usize
    ) -> fn(
        Pointer<Self::MixedType>,
        Pointer<Self::MixedType>,
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
        fn(*mut Self::SelfVec, *const Self::MixedVec),
        fn(&mut Self, &Self::MixedType),
        F,
        G
    ) {
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn get_gemv_kernel() -> fn(
        a: Pointer<Self>,
        b: Pointer<Self>,
        c: Pointer<Self>,
        n: usize,
        k: usize,
        ldb: i64,
        lhs_col_stride: i64
    );

    #[allow(unused_variables)]
    fn get_gemv_kernel_mp() -> fn(
        a: Pointer<Self::MixedType>,
        b: Pointer<Self::MixedType>,
        c: Pointer<Self>,
        n: usize,
        k: usize,
        ldb: i64,
        lhs_col_stride: i64,
        fn(*mut Self::SelfVec, *const Self::MixedVec),
        fn(&mut Self, &Self::MixedType)
    ) {
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn get_gemv_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        F2: Fn(Self::SelfVec, usize, usize) -> Self::SelfVec
    >() -> fn(
        a: Pointer<Self>,
        b: Pointer<Self>,
        c: Pointer<Self>,
        n: usize,
        k: usize,
        ldb: i64,
        lhs_col_stride: i64,
        m_offset: usize,
        n_offset: usize,
        post_op: F,
        post_op_vec: F2
    );

    #[allow(unused_variables)]
    fn get_gemv_kernel_mp_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(Self::SelfVec, usize, usize) -> Self::SelfVec
    >() -> fn(
        a: Pointer<Self::MixedType>,
        b: Pointer<Self::MixedType>,
        c: Pointer<Self>,
        n: usize,
        k: usize,
        ldb: i64,
        lhs_col_stride: i64,
        m_offset: usize,
        n_offset: usize,
        fn(*mut Self::SelfVec, *const Self::MixedVec),
        fn(&mut Self, &Self::MixedType),
        F,
        G
    ) {
        unimplemented!()
    }

    fn get_max_mixed_precision_mr() -> usize {
        unimplemented!()
    }
    fn get_max_mixed_precision_nr() -> usize {
        unimplemented!()
    }
    fn get_max_mr() -> usize;
    fn get_max_nr() -> usize;
    fn get_gemv_nr() -> usize;
    fn get_gemv_mp_nr() -> usize {
        unimplemented!()
    }
}
