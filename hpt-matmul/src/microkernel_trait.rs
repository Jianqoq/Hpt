use crate::Pointer;

/// A trait for microkernels of matrix multiplication
pub trait MatmulMicroKernel<SelfVec, MixedType, MixedVec> where Self: Sized {
    fn get_kernel(
        nr: usize,
        mr: usize
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool);
    fn get_horizontal_kernel(
        nr: usize,
        mr: usize
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool);
    fn get_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(SelfVec, usize, usize) -> SelfVec
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
    fn get_horizontal_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(SelfVec, usize, usize) -> SelfVec
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
    fn get_mixed_precision_kernel(
        nr: usize,
        mr: usize
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
        fn(*mut SelfVec, *const MixedVec),
        fn(&mut Self, &MixedType)
    ) {
        unimplemented!()
    }
    fn get_mixed_precision_kernel_with_post_op<
        F: Fn(Self, usize, usize) -> Self,
        G: Fn(SelfVec, usize, usize) -> SelfVec
    >(
        nr: usize,
        mr: usize
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
        usize,
        usize,
        fn(*mut SelfVec, *const MixedVec),
        fn(&mut Self, &MixedType),
        F,
        G
    ) {
        unimplemented!()
    }
    fn get_max_mixed_precision_mr() -> usize {
        unimplemented!()
    }
    fn get_max_mixed_precision_nr() -> usize{
        unimplemented!()
    }
    fn get_max_mr() -> usize;
    fn get_max_nr() -> usize;
    fn get_horizontal_max_nr() -> usize;
}
