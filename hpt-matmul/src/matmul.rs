use crate::{
    microkernel_trait::MatmulMicroKernel,
    template::matmul_mp,
    utils::kernel_params,
    vec_size,
    Pointer,
    Zero,
};

pub(crate) fn matmul_template_no_block_info<T, TVec>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize
)
    where T: Zero + MatmulMicroKernel<TVec, T, TVec> + Send + Sync + Copy
{
    let (nr, mr) = if m > 1 {
        (T::get_max_nr() * vec_size::<T>(), T::get_max_mr())
    } else {
        (T::get_horizontal_max_nr() * vec_size::<T>(), 1)
    };
    #[cfg(not(target_feature = "neon"))]
    let mut do_lhs_pack = false;
    #[cfg(target_feature = "neon")]
    let mut do_lhs_pack = true;

    if (lhs_col_stride == 1 && n > 128 * nr) || lhs_col_stride != 1 {
        do_lhs_pack = true;
    }
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), do_lhs_pack);
    if param.nc == 0 {
        param.nc = n.next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    super::template::matmul::<T, TVec, _, _>(
        Pointer::new(a as *mut T, 0),
        Pointer::new(b as *mut T, 0),
        Pointer::new(out as *mut T, 0),
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        lhs_col_stride,
        rhs_col_stride,
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        do_lhs_pack,
        num_threads,
        None,
        |_, _, _| unreachable!(),
        |_, _, _| unreachable!()
    );
}

pub(crate) fn matmul_post_op_template_no_block_info<T, TVec, F1, F2>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    post_op: F1,
    post_vec_op: F2
)
    where
        T: Zero + MatmulMicroKernel<TVec, T, TVec> + Send + Sync + Copy,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(TVec, usize, usize) -> TVec + Clone + Send + Sync + 'static
{
    let (nr, mr) = if m > 1 {
        (T::get_max_nr() * vec_size::<T>(), T::get_max_mr())
    } else {
        (T::get_horizontal_max_nr() * vec_size::<T>(), 1)
    };
    #[cfg(not(target_feature = "neon"))]
    let mut do_lhs_pack = false;
    #[cfg(target_feature = "neon")]
    let mut do_lhs_pack = true;

    if (lhs_col_stride == 1 && n > 128 * nr) || lhs_col_stride != 1 {
        do_lhs_pack = true;
    }
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), do_lhs_pack);
    if param.nc == 0 {
        param.nc = n.next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    super::template::matmul_post::<T, TVec, _, _>(
        Pointer::new(a as *mut T, 0),
        Pointer::new(b as *mut T, 0),
        Pointer::new(out as *mut T, 0),
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        lhs_col_stride,
        rhs_col_stride,
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        do_lhs_pack,
        num_threads,
        None,
        post_op,
        post_vec_op
    );
}

#[inline(always)]
pub fn matmul_mp_template_no_block_info<T, TVec, IM, IMVec>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    pack_vec: fn(*mut IMVec, *const TVec, usize),
    pack_vec_exceed: fn(*mut IMVec, usize),
    pack_zero: fn(&mut IM, &T),
    vec_cast_back: fn(*mut TVec, *const IMVec),
    cast_back: fn(&mut T, &IM)
)
    where
        T: MatmulMicroKernel<TVec, IM, IMVec> + Send + Sync + Copy + Zero,
        IM: Send + Sync + Copy + Zero
{
    let nr = T::get_max_mixed_precision_nr() * vec_size::<T>();
    let mr = T::get_max_mixed_precision_mr();
    let mut param = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), true);
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    if param.nc == 0 {
        param.nc = m.next_multiple_of(nr);
    }
    matmul_mp::<T, IM, TVec, IMVec, _, _>(
        Pointer::new(a as *mut T, 0),
        Pointer::new(b as *mut T, 0),
        Pointer::new(out as *mut T, 0),
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        lhs_col_stride,
        rhs_col_stride,
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        num_threads,
        None,
        |_, _, _| unreachable!(),
        |_, _, _| unreachable!(),
        pack_vec,
        pack_vec_exceed,
        pack_zero,
        vec_cast_back,
        cast_back
    );
}

pub fn matmul<T: 'static>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize
) {
    macro_rules! case {
        ($dtype:ty, $vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                matmul_template_no_block_info::<$dtype, $vec>(
                    a as *const $dtype,
                    b as *const $dtype,
                    out as *mut $dtype,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    lhs_col_stride,
                    rhs_col_stride,
                    num_threads
                );
                return;
            }
        };
    }
    case!(bool, crate::BoolVec);
    case!(f32, crate::F32Vec);
    case!(f64, crate::F64Vec);
    case!(i8, crate::I8Vec);
    case!(u8, crate::U8Vec);
    case!(i16, crate::I16Vec);
    case!(u16, crate::U16Vec);
    case!(i32, crate::I32Vec);
    case!(u32, crate::U32Vec);
    case!(i64, crate::I64Vec);
    case!(u64, crate::U64Vec);
    case!(num_complex::Complex32, crate::Cplx32Vec);
    case!(num_complex::Complex64, crate::Cplx64Vec);

    macro_rules! f16_case {
        ($dtype:ty, $vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                matmul_mp_template_no_block_info::<$dtype, $vec, f32, crate::F32Vec>(
                    a as *const $dtype,
                    b as *const $dtype,
                    out as *mut $dtype,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    lhs_col_stride,
                    rhs_col_stride,
                    num_threads,
                    |packed_b, b, i| unsafe {
                        let packed_b = packed_b as *mut crate::F32Vec;
                        let b = b as *const crate::F16Vec;
                        let packed_b_vec0 = packed_b.add(i * 2);
                        let packed_b_vec1 = packed_b.add(i * 2 + 1);
                        let b_vec = b.add(i).read_unaligned();
                        let val_f32 = b_vec.to_2_f32vec();
                        packed_b_vec0.write(val_f32[0]);
                        packed_b_vec1.write(val_f32[1]);
                    },
                    |packed_b, i| unsafe {
                        let packed_b = packed_b as *mut crate::F32Vec;
                        let packed_b_vec0 = packed_b.add(i * 2);
                        let packed_b_vec1 = packed_b.add(i * 2 + 1);
                        packed_b_vec0.write(crate::F32Vec::splat(0.0));
                        packed_b_vec1.write(crate::F32Vec::splat(0.0));
                    },
                    |im, val| {
                        let val = val as *const $dtype;
                        *im = unsafe { val.read().to_f32() };
                    },
                    |im, val| {
                        let im = im as *mut crate::F16Vec;
                        let val = val as *const crate::F32Vec;
                        let vec0 = unsafe { val.read() };
                        let vec1 = unsafe { val.add(1).read() };
                        unsafe { im.write_unaligned(crate::F16Vec::from_2_f32vec([vec0, vec1])) };
                    },
                    |im, val| {
                        let im = im as *mut $dtype;
                        unsafe { *im = <$dtype>::from_f32(*val) };
                    },
                );
            }
        };
    }
    f16_case!(half::f16, crate::F16Vec);
    f16_case!(half::bf16, crate::Bf16Vec);
}

pub fn matmul_with_post<T: 'static, TVec, F1, F2>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize,
    post_op: F1,
    post_vec_op: F2
)
    where
        T: Zero + MatmulMicroKernel<TVec, T, TVec> + Send + Sync + Copy,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(TVec, usize, usize) -> TVec + Clone + Send + Sync + 'static
{
    matmul_post_op_template_no_block_info::<T, TVec, F1, F2>(
        a,
        b,
        out,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        lhs_col_stride,
        rhs_col_stride,
        num_threads,
        post_op,
        post_vec_op
    );
}
