use hpt_common::Pointer;

pub trait MicroKernel: Sized {
    fn get_kernel(
        nr: usize,
        mr: usize,
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool);

    fn get_max_mr() -> usize;
}
