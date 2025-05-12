use crate::utils::PrePackedRhs;
use crate::{
    Pointer, Zero, microkernel_trait::MatmulMicroKernel, template::matmul_mp, utils::kernel_params,
    vec_size,
};
use gemm_common::cache::DivCeil;

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
    num_threads: usize,
    prepacked_rhs: Option<PrePackedRhs>,
) where
    T: Zero + MatmulMicroKernel + Send + Sync + Copy,
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
    let mut param = kernel_params(n, m, k, nr, mr, size_of::<T>(), do_lhs_pack);
    if param.nc == 0 {
        param.nc = n.next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    super::template::matmul::<T, _, _>(
        Pointer::new(a as *mut T, 0),
        Pointer::new(b as *mut T, 0),
        Pointer::new(out, 0),
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
        prepacked_rhs,
        |_, _, _| unreachable!(),
        |_, _, _| unreachable!(),
    );
}

pub(crate) fn matmul_post_op_template_no_block_info<T, F1, F2>(
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
    prepacked_rhs: Option<PrePackedRhs>,
    post_op: F1,
    post_vec_op: F2,
) where
    T: Zero + MatmulMicroKernel + Send + Sync + Copy,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync,
    F2: Fn(<T as MatmulMicroKernel>::SelfVec, usize, usize) -> <T as MatmulMicroKernel>::SelfVec
        + Clone
        + Send
        + Sync,
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
    let mut param = kernel_params(n, m, k, nr, mr, size_of::<T>(), do_lhs_pack);
    if param.nc == 0 {
        param.nc = n.next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    super::template::matmul_post::<T, _, _>(
        Pointer::new(a as *mut T, 0),
        Pointer::new(b as *mut T, 0),
        Pointer::new(out, 0),
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
        prepacked_rhs,
        post_op,
        post_vec_op,
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
    prepacked_rhs: Option<PrePackedRhs>,
    pack_vec: fn(*mut IMVec, *const TVec, usize),
    pack_vec_exceed: fn(*mut IMVec, usize),
    pack_zero: fn(&mut IM, &T),
    vec_cast_back: fn(*mut TVec, *const IMVec),
    cast_back: fn(&mut T, &IM),
) where
    T: MatmulMicroKernel<SelfVec = TVec, MixedType = IM, MixedVec = IMVec>
        + Send
        + Sync
        + Copy
        + Zero,
    IM: Send + Sync + Copy + Zero,
{
    let nr = T::get_max_mixed_precision_nr() * vec_size::<T>();
    let mr = T::get_max_mixed_precision_mr();
    let mut param = kernel_params(n, m, k, nr, mr, size_of::<T>(), true);
    if param.mc == 0 {
        param.mc = m.next_multiple_of(mr);
    }
    if param.nc == 0 {
        param.nc = m.next_multiple_of(nr);
    }
    matmul_mp::<T, _, _>(
        Pointer::new(a as *mut T, 0),
        Pointer::new(b as *mut T, 0),
        Pointer::new(out, 0),
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
        prepacked_rhs,
        |_, _, _| unreachable!(),
        |_, _, _| unreachable!(),
        pack_vec,
        pack_vec_exceed,
        pack_zero,
        vec_cast_back,
        cast_back,
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
    num_threads: usize,
    prepacked_rhs: Option<PrePackedRhs>,
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
                    num_threads,
                    prepacked_rhs,
                );
                return;
            }
        };
    }
    #[cfg(feature = "bool")]
    case!(bool, crate::BoolVec);
    #[cfg(feature = "f32")]
    case!(f32, crate::F32Vec);
    #[cfg(feature = "f64")]
    case!(f64, crate::F64Vec);
    #[cfg(feature = "i8")]
    case!(i8, crate::I8Vec);
    #[cfg(feature = "u8")]
    case!(u8, crate::U8Vec);
    #[cfg(feature = "i16")]
    case!(i16, crate::I16Vec);
    #[cfg(feature = "u16")]
    case!(u16, crate::U16Vec);
    #[cfg(feature = "i32")]
    case!(i32, crate::I32Vec);
    #[cfg(feature = "u32")]
    case!(u32, crate::U32Vec);
    #[cfg(feature = "i64")]
    case!(i64, crate::I64Vec);
    #[cfg(feature = "u64")]
    case!(u64, crate::U64Vec);
    #[cfg(feature = "cplx32")]
    case!(num_complex::Complex32, crate::Cplx32Vec);
    #[cfg(feature = "cplx64")]
    case!(num_complex::Complex64, crate::Cplx64Vec);

    #[cfg(any(feature = "f16", feature = "bf16"))]
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
                    prepacked_rhs,
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
    #[cfg(feature = "f16")]
    f16_case!(half::f16, crate::F16Vec);
    #[cfg(feature = "bf16")]
    f16_case!(half::bf16, crate::Bf16Vec);
}

pub fn matmul_with_post<T: 'static, F1, F2>(
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
    prepacked_rhs: Option<PrePackedRhs>,
    post_op: F1,
    post_vec_op: F2,
) where
    T: Zero + MatmulMicroKernel + Send + Sync + Copy,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync,
    F2: Fn(<T as MatmulMicroKernel>::SelfVec, usize, usize) -> <T as MatmulMicroKernel>::SelfVec
        + Clone
        + Send
        + Sync,
{
    matmul_post_op_template_no_block_info::<T, F1, F2>(
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
        prepacked_rhs,
        post_op,
        post_vec_op,
    );
}

pub(crate) fn prepack_rhs<T>(
    b: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    ldb: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    num_threads: usize,
) -> PrePackedRhs
where
    T: MatmulMicroKernel + Copy,
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
        param.nc = n.msrv_next_multiple_of(nr);
    }
    if param.mc == 0 {
        param.mc = m.msrv_next_multiple_of(mr);
    }
    crate::utils::prepack_b::<T, <T as MatmulMicroKernel>::SelfVec>(
        b,
        m,
        n,
        k,
        ldb,
        rhs_col_stride,
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        num_threads,
    )
}

pub fn matmul_prepack_rhs<T>(
    rhs: *const T,
    len: usize,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    rhs_row_stride: i64,
    lhs_shape: &[i64],
    rhs_shape: &[i64],
    threads: usize,
) -> PrePackedRhs
where
    T: MatmulMicroKernel + Copy,
{
    let m = lhs_shape[lhs_shape.len() - 2] as usize;
    let n = rhs_shape[rhs_shape.len() - 1] as usize;
    let k = rhs_shape[rhs_shape.len() - 2] as usize;

    prepack_rhs(
        Pointer::new(rhs as *mut T, len),
        m,
        n,
        k,
        rhs_row_stride,
        lhs_col_stride,
        rhs_col_stride,
        threads,
    )
}
