use std::sync::Arc;

use crate::utils::{
    prepack_b_mp_single_thread,
    prepack_b_single_thread,
    prepack_lhs,
    prepack_lhs_mp,
    NewPrePackedRhs,
};
use crate::{ Pointer, Zero, microkernel_trait::MatmulMicroKernel, utils::kernel_params, vec_size };
use gemm_common::cache::DivCeil;
use num_integer::Integer;
use rayon::iter::{ IntoParallelIterator, ParallelIterator };

pub(crate) fn _matmul<T, F1, F2>(
    n: usize,
    m: usize,
    k: usize,
    lhs_ptr: (*const T, i64),
    lhs_strides: [i64; 2],
    rhs_ptr: (*const T, i64),
    rhs_strides: [i64; 2],
    res_ptr: (*mut T, i64),
    res_strides: [i64; 2],
    num_threads: usize,
    prepacked_rhs: Option<Arc<Vec<NewPrePackedRhs>>>,
    has_post_op: bool,
    post_op: F1,
    post_op_vec: F2
)
    where
        T: MatmulMicroKernel + Clone + Copy + 'static + Zero + Send + Sync,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(
            <T as MatmulMicroKernel>::SelfVec,
            usize,
            usize
        ) -> <T as MatmulMicroKernel>::SelfVec +
            Clone +
            Send +
            Sync +
            'static
{
    let matmul_fn = if has_post_op {
        crate::template::matmul_post::<T, F1, F2>
    } else {
        crate::template::matmul::<T, F1, F2>
    };
    let lhs_ptr = Pointer::new(lhs_ptr.0 as *mut T, lhs_ptr.1);
    let rhs_ptr = Pointer::new(rhs_ptr.0 as *mut T, rhs_ptr.1);
    let res_ptr = Pointer::new(res_ptr.0 as *mut T, res_ptr.1);
    let nr = <T as MatmulMicroKernel>::get_max_nr() * vec_size::<T>();
    let mr = <T as MatmulMicroKernel>::get_max_mr();
    #[cfg(not(target_feature = "neon"))]
    let mut do_lhs_pack = false;
    #[cfg(target_feature = "neon")]
    let mut do_lhs_pack = true;

    if (lhs_strides[1] == 1 && n > 128 * nr) || lhs_strides[1] != 1 {
        do_lhs_pack = true;
    }
    let mut params = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), do_lhs_pack);
    if params.nc == 0 {
        params.nc = n.msrv_next_multiple_of(nr);
    }
    if params.mc == 0 {
        params.mc = m.msrv_next_multiple_of(mr);
    }
    let kc = params.kc;
    let mut mc = params.mc;
    let mut nc = params.nc;
    if n > m {
        let func = |nc: usize| {
            let num_nc = n.div_ceil(nc);
            let lda = lhs_strides[0];
            let lhs_col_stride = lhs_strides[1];
            let prepacked_lhs = prepack_lhs(
                #[cfg(feature = "bound_check")] (lhs_ptr.ptr, lhs_ptr.len),
                #[cfg(not(feature = "bound_check"))] lhs_ptr.ptr,
                lhs_strides,
                mr,
                mc,
                kc,
                k,
                m
            );
            let execute = move |i: usize| {
                let start = i * nc;
                let end = (start + nc).min(n);
                let rhs_ptr = rhs_ptr + (start as i64) * (rhs_strides[1] as i64);
                let res_ptr = res_ptr + (start as i64) * (res_strides[1] as i64);
                let lhs_ptr = lhs_ptr;
                let sliced_m = m as i64;
                let sliced_k = k as i64;
                let sliced_n = end - start;
                let sliced_ldb = rhs_strides[0];
                let sliced_ldc = res_strides[0];
                let sliced_rhs_col_stride = rhs_strides[1];
                let prepacked_rhs = if let Some(prepacked_rhs) = &prepacked_rhs {
                    Some(&prepacked_rhs[i])
                } else {
                    None
                };
                matmul_fn(
                    lhs_ptr,
                    rhs_ptr,
                    res_ptr,
                    0,
                    start,
                    sliced_m as usize,
                    sliced_n as usize,
                    sliced_k as usize,
                    lda,
                    sliced_ldb,
                    sliced_ldc,
                    lhs_col_stride,
                    sliced_rhs_col_stride,
                    kc,
                    mc,
                    nc,
                    nr,
                    mr,
                    do_lhs_pack,
                    Some(&prepacked_lhs),
                    prepacked_rhs,
                    post_op.clone(),
                    post_op_vec.clone()
                );
            };
            if num_threads == 1 {
                for i in 0..num_nc {
                    execute(i);
                }
            } else {
                (0..num_nc).into_par_iter().for_each(move |i| {
                    execute(i);
                });
            }
        };
        let num_nc = n.div_ceil(nc);
        if num_nc >= num_threads {
            func(nc);
        } else {
            while nc != 0 && n.div_ceil(nc) < num_threads {
                nc = (nc - 1).prev_multiple_of(&nr);
            }
            if nc == 0 {
                func(params.nc);
            } else {
                func(nc);
            }
        }
    } else {
        let func = |mc: usize| {
            let num_mc = m.div_ceil(mc);
            let holder;
            let prepacked_rhs = if let Some(prepacked_rhs) = &prepacked_rhs {
                &prepacked_rhs[0]
            } else {
                let (ptrs, packed_b, layout) = prepack_b_single_thread::<T>(
                    rhs_ptr,
                    n,
                    k,
                    rhs_strides[0],
                    rhs_strides[1],
                    kc,
                    n,
                    nr,
                    true
                );
                let prepacked_rhs = NewPrePackedRhs {
                    buffers: ptrs,
                    buffer: (packed_b, layout),
                    nr,
                    nc,
                    kc,
                };
                holder = prepacked_rhs;
                &holder
            };
            let execute = move |i: usize| {
                let start = i * mc;
                let end = (start + mc).min(m);
                let rhs_ptr = rhs_ptr;
                let res_ptr = res_ptr + (start as i64) * (res_strides[0] as i64);
                let lhs_ptr = lhs_ptr + (start as i64) * (lhs_strides[0] as i64);
                let sliced_m = end - start;
                let sliced_k = k as i64;
                let sliced_n = n as i64;
                let sliced_lda = lhs_strides[0];
                let sliced_ldb = rhs_strides[0];
                let sliced_ldc = res_strides[0];
                let sliced_lhs_col_stride = lhs_strides[1];
                let sliced_rhs_col_stride = rhs_strides[1];
                matmul_fn(
                    lhs_ptr,
                    rhs_ptr,
                    res_ptr,
                    start,
                    0,
                    sliced_m as usize,
                    sliced_n as usize,
                    sliced_k as usize,
                    sliced_lda,
                    sliced_ldb,
                    sliced_ldc,
                    sliced_lhs_col_stride,
                    sliced_rhs_col_stride,
                    kc,
                    mc,
                    nc,
                    nr,
                    mr,
                    do_lhs_pack,
                    None,
                    Some(&prepacked_rhs),
                    post_op.clone(),
                    post_op_vec.clone()
                );
            };
            if num_threads == 1 {
                for i in 0..num_mc {
                    execute(i);
                }
            } else {
                (0..num_mc).into_par_iter().for_each(move |i| {
                    execute(i);
                });
            }
        };
        let num_mc = m.div_ceil(params.mc);
        if num_mc >= num_threads {
            func(params.mc);
        } else {
            while mc != 0 && m.div_ceil(mc) < num_threads {
                mc = (mc - 1).prev_multiple_of(&mr);
            }
            if mc == 0 {
                func(params.mc);
            } else {
                func(mc);
            }
        }
    }
}

pub(crate) fn _matmul_mp<T, F1, F2>(
    n: usize,
    m: usize,
    k: usize,
    lhs_ptr: (*const T, i64),
    lhs_strides: [i64; 2],
    rhs_ptr: (*const T, i64),
    rhs_strides: [i64; 2],
    res_ptr: (*mut T, i64),
    res_strides: [i64; 2],
    num_threads: usize,
    prepacked_rhs: Option<Arc<Vec<NewPrePackedRhs>>>,
    has_post_op: bool,
    post_op: F1,
    post_op_vec: F2,
    pack_vec: fn(
        *mut <T as MatmulMicroKernel>::MixedVec,
        *const <T as MatmulMicroKernel>::SelfVec,
        usize
    ),
    pack_vec_exceed: fn(*mut <T as MatmulMicroKernel>::MixedVec, usize),
    pack_zero: fn(&mut <T as MatmulMicroKernel>::MixedType, &T),
    vec_cast_back: fn(
        *mut <T as MatmulMicroKernel>::SelfVec,
        *const <T as MatmulMicroKernel>::MixedVec
    ),
    cast_back: fn(&mut T, &<T as MatmulMicroKernel>::MixedType)
)
    where
        T: MatmulMicroKernel + Clone + Copy + 'static + Zero + Send + Sync,
        <T as MatmulMicroKernel>::MixedType: Copy + Zero,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(
            <T as MatmulMicroKernel>::SelfVec,
            usize,
            usize
        ) -> <T as MatmulMicroKernel>::SelfVec +
            Clone +
            Send +
            Sync +
            'static
{
    let matmul_fn = if has_post_op {
        crate::template::matmul_post_mp_sg::<T, F1, F2>
    } else {
        crate::template::matmul_mp_sg::<T, F1, F2>
    };
    let lhs_ptr = Pointer::new(lhs_ptr.0 as *mut T, lhs_ptr.1);
    let rhs_ptr = Pointer::new(rhs_ptr.0 as *mut T, rhs_ptr.1);
    let res_ptr = Pointer::new(res_ptr.0 as *mut T, res_ptr.1);
    let nr = <T as MatmulMicroKernel>::get_max_mixed_precision_nr() * vec_size::<T>();
    let mr = <T as MatmulMicroKernel>::get_max_mixed_precision_mr();
    let do_lhs_pack = true;
    let mut params = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), do_lhs_pack);
    if params.nc == 0 {
        params.nc = n.msrv_next_multiple_of(nr);
    }
    if params.mc == 0 {
        params.mc = m.msrv_next_multiple_of(mr);
    }
    let kc = params.kc;
    let mut mc = params.mc;
    let mut nc = params.nc;
    if n > m {
        let func = |nc: usize| {
            let num_nc = n.div_ceil(nc);
            let lda = lhs_strides[0];
            let lhs_col_stride = lhs_strides[1];
            let prepacked_lhs = prepack_lhs_mp(
                #[cfg(feature = "bound_check")] (lhs_ptr.ptr, lhs_ptr.len),
                #[cfg(not(feature = "bound_check"))] lhs_ptr.ptr,
                lhs_strides,
                mr,
                mc,
                kc,
                k,
                m,
                pack_zero
            );
            let execute = move |i: usize| {
                let start = i * nc;
                let end = (start + nc).min(n);
                let rhs_ptr = rhs_ptr + (start as i64) * (rhs_strides[1] as i64);
                let res_ptr = res_ptr + (start as i64) * (res_strides[1] as i64);
                let lhs_ptr = lhs_ptr;
                let sliced_m = m as i64;
                let sliced_k = k as i64;
                let sliced_n = end - start;
                let sliced_ldb = rhs_strides[0];
                let sliced_ldc = res_strides[0];
                let sliced_rhs_col_stride = rhs_strides[1];
                let prepacked_rhs = if let Some(prepacked_rhs) = &prepacked_rhs {
                    Some(&prepacked_rhs[i])
                } else {
                    None
                };
                matmul_fn(
                    lhs_ptr,
                    rhs_ptr,
                    res_ptr,
                    0,
                    start,
                    sliced_m as usize,
                    sliced_n as usize,
                    sliced_k as usize,
                    lda,
                    sliced_ldb,
                    sliced_ldc,
                    lhs_col_stride,
                    sliced_rhs_col_stride,
                    kc,
                    mc,
                    nc,
                    nr,
                    mr,
                    do_lhs_pack,
                    Some(&prepacked_lhs),
                    prepacked_rhs,
                    post_op.clone(),
                    post_op_vec.clone(),
                    pack_vec,
                    pack_vec_exceed,
                    pack_zero,
                    vec_cast_back,
                    cast_back
                );
            };
            if num_threads == 1 {
                for i in 0..num_nc {
                    execute(i);
                }
            } else {
                (0..num_nc).into_par_iter().for_each(move |i| {
                    execute(i);
                });
            }
        };
        let num_nc = n.div_ceil(nc);
        if num_nc >= num_threads {
            func(nc);
        } else {
            while nc != 0 && n.div_ceil(nc) < num_threads {
                nc = (nc - 1).prev_multiple_of(&nr);
            }
            if nc == 0 {
                func(params.nc);
            } else {
                func(nc);
            }
        }
    } else {
        let func = |mc: usize| {
            let num_mc = m.div_ceil(mc);
            let holder;
            let prepacked_rhs = if let Some(prepacked_rhs) = &prepacked_rhs {
                &prepacked_rhs[0]
            } else {
                let (ptrs, packed_b, layout) = prepack_b_mp_single_thread::<T>(
                    rhs_ptr,
                    n,
                    k,
                    rhs_strides[0],
                    rhs_strides[1],
                    kc,
                    n,
                    nr,
                    true,
                    pack_vec,
                    pack_vec_exceed,
                    pack_zero
                );
                let prepacked_rhs = NewPrePackedRhs {
                    buffers: ptrs,
                    buffer: (packed_b, layout),
                    nr,
                    nc,
                    kc,
                };
                holder = prepacked_rhs;
                &holder
            };
            let execute = move |i: usize| {
                let start = i * mc;
                let end = (start + mc).min(m);
                let rhs_ptr = rhs_ptr;
                let res_ptr = res_ptr + (start as i64) * (res_strides[0] as i64);
                let lhs_ptr = lhs_ptr + (start as i64) * (lhs_strides[0] as i64);
                let sliced_m = end - start;
                let sliced_k = k as i64;
                let sliced_n = n as i64;
                let sliced_lda = lhs_strides[0];
                let sliced_ldb = rhs_strides[0];
                let sliced_ldc = res_strides[0];
                let sliced_lhs_col_stride = lhs_strides[1];
                let sliced_rhs_col_stride = rhs_strides[1];
                matmul_fn(
                    lhs_ptr,
                    rhs_ptr,
                    res_ptr,
                    start,
                    0,
                    sliced_m as usize,
                    sliced_n as usize,
                    sliced_k as usize,
                    sliced_lda,
                    sliced_ldb,
                    sliced_ldc,
                    sliced_lhs_col_stride,
                    sliced_rhs_col_stride,
                    kc,
                    mc,
                    nc,
                    nr,
                    mr,
                    do_lhs_pack,
                    None,
                    Some(&prepacked_rhs),
                    post_op.clone(),
                    post_op_vec.clone(),
                    pack_vec,
                    pack_vec_exceed,
                    pack_zero,
                    vec_cast_back,
                    cast_back
                );
            };
            if num_threads == 1 {
                for i in 0..num_mc {
                    execute(i);
                }
            } else {
                (0..num_mc).into_par_iter().for_each(move |i| {
                    execute(i);
                });
            }
        };
        let num_mc = m.div_ceil(params.mc);
        if num_mc >= num_threads {
            func(params.mc);
        } else {
            while mc != 0 && m.div_ceil(mc) < num_threads {
                mc = (mc - 1).prev_multiple_of(&mr);
            }
            if mc == 0 {
                func(params.mc);
            } else {
                func(mc);
            }
        }
    }
}

pub fn matmul<T: 'static>(
    (lhs_ptr, lhs_len): (*const T, i64),
    (rhs_ptr, rhs_len): (*const T, i64),
    (res_ptr, res_len): (*mut T, i64),
    n: usize,
    m: usize,
    k: usize,
    lhs_strides: [i64; 2],
    rhs_strides: [i64; 2],
    res_strides: [i64; 2],
    num_threads: usize,
    prepacked_rhs: Option<Arc<Vec<NewPrePackedRhs>>>,
    post_op: Option<&(dyn (Fn(T, usize, usize) -> T) + Send + Sync + 'static)>,
    post_op_vec: Option<
        &(dyn (Fn(
            <T as MatmulMicroKernel>::SelfVec,
            usize,
            usize
        ) -> <T as MatmulMicroKernel>::SelfVec) +
            Send +
            Sync +
            'static)
    >
)
    where T: MatmulMicroKernel
{
    macro_rules! case {
        ($dtype:ty, $vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                match (post_op, post_op_vec) {
                    (Some(post_op), Some(post_op_vec)) => {
                        let post_op: &(dyn (Fn($dtype, usize, usize) -> $dtype) + Send + Sync + 'static) = unsafe {
                            std::mem::transmute(post_op)
                        };
                        let post_op_vec: &(dyn (Fn($vec, usize, usize) -> $vec) +
                            Send +
                            Sync +
                            'static) = unsafe { std::mem::transmute(post_op_vec) };
                        _matmul::<$dtype, _, _>(
                            n,
                            m,
                            k,
                            (lhs_ptr as *const $dtype, lhs_len),
                            lhs_strides,
                            (rhs_ptr as *const $dtype, rhs_len),
                            rhs_strides,
                            (res_ptr as *mut $dtype, res_len),
                            res_strides,
                            num_threads,
                            prepacked_rhs,
                            true,
                            post_op,
                            post_op_vec
                        );
                    }
                    _ => {
                        _matmul::<$dtype, _, _>(
                            n,
                            m,
                            k,
                            (lhs_ptr as *const $dtype, lhs_len),
                            lhs_strides,
                            (rhs_ptr as *const $dtype, rhs_len),
                            rhs_strides,
                            (res_ptr as *mut $dtype, res_len),
                            res_strides,
                            num_threads,
                            prepacked_rhs,
                            false,
                            |_, _, _| unreachable!(),
                            |_, _, _| unreachable!()
                        );
                    }
                }
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
        ($dtype:ty, $vec:ty, $im:ty, $im_vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                type T = $dtype;
                type Vec = $vec;
                type IM = $im;
                type IMVec = $im_vec;
                fn pack_vec(packed_b: *mut IMVec, b: *const Vec, i: usize) {
                    let packed_b = packed_b as *mut IMVec;
                    let b = b as *const Vec;
                    unsafe {
                        let packed_b_vec0 = packed_b.add(i * 2);
                        let packed_b_vec1 = packed_b.add(i * 2 + 1);
                        let b_vec = b.add(i).read_unaligned();
                        let val_f32 = b_vec.to_2_f32vec();
                        packed_b_vec0.write(val_f32[0]);
                        packed_b_vec1.write(val_f32[1]);
                    }
                }
                fn pack_vec_exceed(packed_b: *mut IMVec, i: usize) {
                    unsafe {
                        let packed_b = packed_b as *mut IMVec;
                        let packed_b_vec0 = packed_b.add(i * 2);
                        let packed_b_vec1 = packed_b.add(i * 2 + 1);
                        packed_b_vec0.write(IMVec::splat(0.0));
                        packed_b_vec1.write(IMVec::splat(0.0));
                    }
                }
                fn pack_zero(im: &mut IM, val: &T) {
                    let val = val as *const T;
                    *im = unsafe { val.read().to_f32() };
                }
                fn vec_cast_back(im: *mut Vec, val: *const IMVec) {
                    let im = im as *mut Vec;
                    let val = val as *const IMVec;
                    let vec0 = unsafe { val.read() };
                    let vec1 = unsafe { val.add(1).read() };
                    unsafe { im.write_unaligned(Vec::from_2_f32vec([vec0, vec1])) };
                }
                fn cast_back(im: &mut T, val: &IM) {
                    let im = im as *mut T;
                    unsafe { *im = <T>::from_f32(*val) };
                }
                
                match (post_op, post_op_vec) {
                    (Some(post_op), Some(post_op_vec)) => {
                        let post_op: &(dyn (Fn(T, usize, usize) -> T) + Send + Sync + 'static) = unsafe {
                            std::mem::transmute(post_op)
                        };
                        let post_op_vec: &(dyn (Fn(Vec, usize, usize) -> Vec) +
                            Send +
                            Sync +
                            'static) = unsafe { std::mem::transmute(post_op_vec) };
                        _matmul_mp::<T, _, _>(
                            n,
                            m,
                            k,
                            (lhs_ptr as *const T, lhs_len),
                            lhs_strides,
                            (rhs_ptr as *const T, rhs_len),
                            rhs_strides,
                            (res_ptr as *mut T, res_len),
                            res_strides,
                            num_threads,
                            prepacked_rhs,
                            true,
                            post_op,
                            post_op_vec,
                            pack_vec,
                            pack_vec_exceed,
                            pack_zero,
                            vec_cast_back,
                            cast_back,
                        );
                    }
                    _ => {
                        _matmul_mp::<T, _, _>(
                            n,
                            m,
                            k,
                            (lhs_ptr as *const T, lhs_len),
                            lhs_strides,
                            (rhs_ptr as *const T, rhs_len),
                            rhs_strides,
                            (res_ptr as *mut T, res_len),
                            res_strides,
                            num_threads,
                            prepacked_rhs,
                            false,
                            |_, _, _| unreachable!(),
                            |_, _, _| unreachable!(),
                            pack_vec,
                            pack_vec_exceed,
                            pack_zero,
                            vec_cast_back,
                            cast_back,
                        );
                    }
                }
                return;
            }
        };
    }

    #[cfg(feature = "f16")]
    f16_case!(half::f16, crate::F16Vec, f32, crate::F32Vec);
    #[cfg(feature = "bf16")]
    f16_case!(half::bf16, crate::Bf16Vec, f32, crate::F32Vec);
}

pub fn addmm<T: 'static>(
    (lhs_ptr, lhs_len): (*const T, i64),
    (rhs_ptr, rhs_len): (*const T, i64),
    (res_ptr, res_len): (*mut T, i64),
    (bias_ptr, bias_len): (*const T, i64),
    n: usize,
    m: usize,
    k: usize,
    lhs_strides: [i64; 2],
    rhs_strides: [i64; 2],
    res_strides: [i64; 2],
    bias_strides: [i64; 2],
    num_threads: usize,
    prepacked_rhs: Option<Arc<Vec<NewPrePackedRhs>>>,
    post_op: Option<&(dyn (Fn(T, usize, usize) -> T) + Send + Sync + 'static)>,
    post_op_vec: Option<
        &(dyn (Fn(
            <T as MatmulMicroKernel>::SelfVec,
            usize,
            usize
        ) -> <T as MatmulMicroKernel>::SelfVec) +
            Send +
            Sync +
            'static)
    >
)
    where T: MatmulMicroKernel
{
    macro_rules! case {
        ($dtype:ty, $vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                let bias_ptr = Pointer::new(bias_ptr as *mut $dtype, bias_len);
                match (post_op, post_op_vec) {
                    (Some(post_op), Some(post_op_vec)) => {
                        let post_op: &(dyn (Fn($dtype, usize, usize) -> $dtype) + Send + Sync + 'static) = unsafe {
                            std::mem::transmute(post_op)
                        };
                        let post_op_vec: &(dyn (Fn($vec, usize, usize) -> $vec) +
                            Send +
                            Sync +
                            'static) = unsafe { std::mem::transmute(post_op_vec) };
                        _matmul::<$dtype, _, _>(
                            n,
                            m,
                            k,
                            (lhs_ptr as *const $dtype, lhs_len),
                            lhs_strides,
                            (rhs_ptr as *const $dtype, rhs_len),
                            rhs_strides,
                            (res_ptr as *mut $dtype, res_len),
                            res_strides,
                            num_threads,
                            prepacked_rhs,
                            true,
                            move |x, m, n| {
                                use std::ops::Add;
                                let offset = m as i64 * bias_strides[0] + n as i64 * bias_strides[1];
                                let bias = bias_ptr[offset];
                                post_op(x.add(bias), m, n)
                            },
                            move |x, m, n| {
                                use std::ops::Add;
                                let offset = m as i64 * bias_strides[0] + n as i64 * bias_strides[1];
                                let bias = bias_ptr.offset(offset).cast::<$vec>().read_unaligned();
                                post_op_vec(x.add(bias), m, n)
                            }
                        );
                    }
                    _ => {
                        _matmul::<$dtype, _, _>(
                            n,
                            m,
                            k,
                            (lhs_ptr as *const $dtype, lhs_len),
                            lhs_strides,
                            (rhs_ptr as *const $dtype, rhs_len),
                            rhs_strides,
                            (res_ptr as *mut $dtype, res_len),
                            res_strides,
                            num_threads,
                            prepacked_rhs,
                            true,
                            move |x, m, n| {
                                use std::ops::Add;
                                let offset = m as i64 * bias_strides[0] + n as i64 * bias_strides[1];
                                let bias = bias_ptr[offset];
                                x.add(bias)
                            },
                            move |x, m, n| {
                                use std::ops::Add;
                                let offset = m as i64 * bias_strides[0] + n as i64 * bias_strides[1];
                                let bias = bias_ptr.offset(offset).cast::<$vec>().read_unaligned();
                                x.add(bias)
                            }
                        );
                    }
                }
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
        ($dtype:ty, $vec:ty, $im:ty, $im_vec:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                type T = $dtype;
                type Vec = $vec;
                type IM = $im;
                type IMVec = $im_vec;
                let bias_ptr = Pointer::new(bias_ptr as *mut T, bias_len);
                fn pack_vec(packed_b: *mut IMVec, b: *const Vec, i: usize) {
                    let packed_b = packed_b as *mut IMVec;
                    let b = b as *const Vec;
                    unsafe {
                        let packed_b_vec0 = packed_b.add(i * 2);
                        let packed_b_vec1 = packed_b.add(i * 2 + 1);
                        let b_vec = b.add(i).read_unaligned();
                        let val_f32 = b_vec.to_2_f32vec();
                        packed_b_vec0.write(val_f32[0]);
                        packed_b_vec1.write(val_f32[1]);
                    }
                }
                fn pack_vec_exceed(packed_b: *mut IMVec, i: usize) {
                    unsafe {
                        let packed_b = packed_b as *mut IMVec;
                        let packed_b_vec0 = packed_b.add(i * 2);
                        let packed_b_vec1 = packed_b.add(i * 2 + 1);
                        packed_b_vec0.write(IMVec::splat(0.0));
                        packed_b_vec1.write(IMVec::splat(0.0));
                    }
                }
                fn pack_zero(im: &mut IM, val: &T) {
                    *im = (*val).to_f32();
                }
                fn vec_cast_back(im: *mut Vec, val: *const IMVec) {
                    let im = im as *mut Vec;
                    let val = val as *const IMVec;
                    let vec0 = unsafe { val.read() };
                    let vec1 = unsafe { val.add(1).read() };
                    unsafe { im.write_unaligned(Vec::from_2_f32vec([vec0, vec1])) }
                }
                fn cast_back(im: &mut T, val: &IM) {
                    *im = <T>::from_f32(*val);
                }
                match (post_op, post_op_vec) {
                    (Some(post_op), Some(post_op_vec)) => {
                        let post_op: &(dyn (Fn(T, usize, usize) -> T) + Send + Sync + 'static) = unsafe {
                            std::mem::transmute(post_op)
                        };
                        let post_op_vec: &(dyn (Fn(Vec, usize, usize) -> Vec) +
                            Send +
                            Sync +
                            'static) = unsafe { std::mem::transmute(post_op_vec) };
                        _matmul_mp::<T, _, _>(
                            n,
                            m,
                            k,
                            (lhs_ptr as *const T, lhs_len),
                            lhs_strides,
                            (rhs_ptr as *const T, rhs_len),
                            rhs_strides,
                            (res_ptr as *mut T, res_len),
                            res_strides,
                            num_threads,
                            prepacked_rhs,
                            true,
                            move |x, m, n| {
                                use std::ops::Add;
                                let offset = m as i64 * bias_strides[0] + n as i64 * bias_strides[1];
                                let bias = bias_ptr[offset];
                                post_op(x.add(bias), m, n)
                            },
                            move |x, m, n| {
                                use std::ops::Add;
                                let offset = m as i64 * bias_strides[0] + n as i64 * bias_strides[1];
                                let bias = bias_ptr.offset(offset).cast::<Vec>().read_unaligned();
                                post_op_vec(x.add(bias), m, n)
                            },
                            pack_vec,
                            pack_vec_exceed,
                            pack_zero,
                            vec_cast_back,
                            cast_back
                        );
                    }
                    _ => {
                        _matmul_mp::<T, _, _>(
                            n,
                            m,
                            k,
                            (lhs_ptr as *const T, lhs_len),
                            lhs_strides,
                            (rhs_ptr as *const T, rhs_len),
                            rhs_strides,
                            (res_ptr as *mut T, res_len),
                            res_strides,
                            num_threads,
                            prepacked_rhs,
                            true,
                            move |x, m, n| {
                                use std::ops::Add;
                                let offset = m as i64 * bias_strides[0] + n as i64 * bias_strides[1];
                                let bias = bias_ptr[offset];
                                x.add(bias)
                            },
                            move |x, m, n| {
                                use std::ops::Add;
                                let offset = m as i64 * bias_strides[0] + n as i64 * bias_strides[1];
                                let bias = bias_ptr.offset(offset).cast::<Vec>().read_unaligned();
                                x.add(bias)
                            },
                            pack_vec,
                            pack_vec_exceed,
                            pack_zero,
                            vec_cast_back,
                            cast_back
                        );
                    }
                }
                return;
            }
        };
    }

    #[cfg(feature = "f16")]
    f16_case!(half::f16, crate::F16Vec, f32, crate::F32Vec);
    #[cfg(feature = "bf16")]
    f16_case!(half::bf16, crate::Bf16Vec, f32, crate::F32Vec);

    panic!("Unsupported type: {:?}", std::any::TypeId::of::<T>());
}

pub(crate) fn prepack_rhs_non_mp<T>(
    n: usize,
    m: usize,
    k: usize,
    rhs_ptr: *const T,
    rhs_strides: [i64; 2],
    lhs_col_stride: i64,
    num_threads: usize
) -> Vec<NewPrePackedRhs>
    where T: MatmulMicroKernel + Clone + Copy + 'static + Zero + Send + Sync
{
    let rhs_ptr = Pointer::new(rhs_ptr as *mut T, (n * k) as i64);
    let nr = <T as MatmulMicroKernel>::get_max_nr() * vec_size::<T>();
    let mr = <T as MatmulMicroKernel>::get_max_mr();
    #[cfg(not(target_feature = "neon"))]
    let mut do_lhs_pack = false;
    #[cfg(target_feature = "neon")]
    let mut do_lhs_pack = true;

    if (lhs_col_stride == 1 && n > 128 * nr) || lhs_col_stride != 1 {
        do_lhs_pack = true;
    }
    let mut params = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), do_lhs_pack);
    if params.nc == 0 {
        params.nc = n.msrv_next_multiple_of(nr);
    }
    if params.mc == 0 {
        params.mc = m.msrv_next_multiple_of(mr);
    }
    let kc = params.kc;
    let mut nc = params.nc;

    if n > m {
        let func = |nc: usize| {
            let num_nc = n.div_ceil(nc);

            let res = (0..num_nc)
                .into_par_iter()
                .map(move |i| {
                    let start = i * nc;
                    let end = (start + nc).min(n);
                    let rhs_ptr = rhs_ptr + (start as i64) * rhs_strides[1];
                    let sliced_k = k;
                    let sliced_n = end - start;
                    let sliced_ldb = rhs_strides[0];
                    let sliced_rhs_col_stride = rhs_strides[1];
                    let (ptrs, packed_b, layout) = crate::utils::prepack_b_single_thread::<T>(
                        rhs_ptr,
                        sliced_n,
                        sliced_k,
                        sliced_ldb,
                        sliced_rhs_col_stride,
                        kc,
                        nc,
                        nr,
                        false
                    );
                    let prepacked_rhs = NewPrePackedRhs {
                        buffers: ptrs,
                        buffer: (packed_b, layout),
                        nr,
                        nc,
                        kc,
                    };
                    prepacked_rhs
                })
                .collect::<Vec<_>>();
            res
        };
        let num_nc = n.div_ceil(nc);
        if num_nc >= num_threads {
            func(nc)
        } else {
            while nc != 0 && n.div_ceil(nc) < num_threads {
                nc = (nc - 1).prev_multiple_of(&nr);
            }
            if nc == 0 {
                func(params.nc)
            } else {
                func(nc)
            }
        }
    } else {
        let (ptrs, packed_b, layout) = crate::utils::prepack_b_single_thread::<T>(
            rhs_ptr,
            n,
            k,
            rhs_strides[0],
            rhs_strides[1],
            kc,
            nc,
            nr,
            true
        );
        let prepacked_rhs = NewPrePackedRhs {
            buffers: ptrs,
            buffer: (packed_b, layout),
            nr,
            nc,
            kc,
        };
        vec![prepacked_rhs]
    }
}

pub(crate) fn prepack_rhs_mp<T>(
    n: usize,
    m: usize,
    k: usize,
    rhs_ptr: *const T,
    rhs_strides: [i64; 2],
    lhs_col_stride: i64,
    num_threads: usize,
    pack_vec: fn(
        *mut <T as MatmulMicroKernel>::MixedVec,
        *const <T as MatmulMicroKernel>::SelfVec,
        usize
    ),
    pack_vec_exceed: fn(*mut <T as MatmulMicroKernel>::MixedVec, usize),
    pack_zero: fn(&mut <T as MatmulMicroKernel>::MixedType, &T)
)
    -> Vec<NewPrePackedRhs>
    where
        T: MatmulMicroKernel + Clone + Copy + 'static + Zero + Send + Sync,
        <T as MatmulMicroKernel>::MixedType: Copy + Zero
{
    let rhs_ptr = Pointer::new(rhs_ptr as *mut T, (n * k) as i64);
    let nr = <T as MatmulMicroKernel>::get_max_nr() * vec_size::<T>();
    let mr = <T as MatmulMicroKernel>::get_max_mr();
    #[cfg(not(target_feature = "neon"))]
    let mut do_lhs_pack = false;
    #[cfg(target_feature = "neon")]
    let mut do_lhs_pack = true;

    if (lhs_col_stride == 1 && n > 128 * nr) || lhs_col_stride != 1 {
        do_lhs_pack = true;
    }
    let mut params = kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), do_lhs_pack);
    if params.nc == 0 {
        params.nc = n.msrv_next_multiple_of(nr);
    }
    if params.mc == 0 {
        params.mc = m.msrv_next_multiple_of(mr);
    }
    let kc = params.kc;
    let mut nc = params.nc;

    if n > m {
        let func = |nc: usize| {
            let num_nc = n.div_ceil(nc);

            let res = (0..num_nc)
                .into_par_iter()
                .map(move |i| {
                    let start = i * nc;
                    let end = (start + nc).min(n);
                    let rhs_ptr = rhs_ptr + (start as i64) * rhs_strides[1];
                    let sliced_k = k;
                    let sliced_n = end - start;
                    let sliced_ldb = rhs_strides[0];
                    let sliced_rhs_col_stride = rhs_strides[1];
                    let (ptrs, packed_b, layout) = crate::utils::prepack_b_mp_single_thread::<T>(
                        rhs_ptr,
                        sliced_n,
                        sliced_k,
                        sliced_ldb,
                        sliced_rhs_col_stride,
                        kc,
                        nc,
                        nr,
                        false,
                        pack_vec,
                        pack_vec_exceed,
                        pack_zero
                    );
                    let prepacked_rhs = NewPrePackedRhs {
                        buffers: ptrs,
                        buffer: (packed_b, layout),
                        nr,
                        nc,
                        kc,
                    };
                    prepacked_rhs
                })
                .collect::<Vec<_>>();
            res
        };
        let num_nc = n.div_ceil(nc);
        if num_nc >= num_threads {
            func(nc)
        } else {
            while nc != 0 && n.div_ceil(nc) < num_threads {
                nc = (nc - 1).prev_multiple_of(&nr);
            }
            if nc == 0 {
                func(params.nc)
            } else {
                func(nc)
            }
        }
    } else {
        let (ptrs, packed_b, layout) = crate::utils::prepack_b_mp_single_thread::<T>(
            rhs_ptr,
            n,
            k,
            rhs_strides[0],
            rhs_strides[1],
            kc,
            nc,
            nr,
            true,
            pack_vec,
            pack_vec_exceed,
            pack_zero
        );
        let prepacked_rhs = NewPrePackedRhs {
            buffers: ptrs,
            buffer: (packed_b, layout),
            nr,
            nc,
            kc,
        };
        vec![prepacked_rhs]
    }
}

pub fn prepack_rhs<T>(
    n: usize,
    m: usize,
    k: usize,
    rhs_ptr: *const T,
    rhs_strides: [i64; 2],
    lhs_col_stride: i64,
    num_threads: usize
) -> Vec<NewPrePackedRhs>
    where T: MatmulMicroKernel + Clone + Copy + 'static + Zero + Send + Sync
{
    let is_mp =
        std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::f16>() ||
        std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>();
    if is_mp {
        macro_rules! f16_prepack {
            ($dtype:ty, $vec:ty, $im:ty, $im_vec:ty) => {
                {
                    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$dtype>() {
                        type T = $dtype;
                        type Vec = $vec;
                        type IM = $im;
                        type IMVec = $im_vec;
                        fn pack_vec(packed_b: *mut IMVec, b: *const Vec, i: usize) {
                            let packed_b = packed_b as *mut IMVec;
                            let b = b as *const Vec;
                            unsafe {
                                let packed_b_vec0 = packed_b.add(i * 2);
                                let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                let b_vec = b.add(i).read_unaligned();
                                let val_f32 = b_vec.to_2_f32vec();
                                packed_b_vec0.write(val_f32[0]);
                                packed_b_vec1.write(val_f32[1]);
                            }
                        }
                        fn pack_vec_exceed(packed_b: *mut IMVec, i: usize) {
                            unsafe {
                                let packed_b = packed_b as *mut IMVec;
                                let packed_b_vec0 = packed_b.add(i * 2);
                                let packed_b_vec1 = packed_b.add(i * 2 + 1);
                                packed_b_vec0.write(IMVec::splat(0.0));
                                packed_b_vec1.write(IMVec::splat(0.0));
                            }
                        }
                        fn pack_zero(im: &mut IM, val: &T) {
                            *im = (*val).to_f32();
                        }
                        return prepack_rhs_mp::<T>(
                            n,
                            m,
                            k,
                            rhs_ptr as *const T,
                            rhs_strides,
                            lhs_col_stride,
                            num_threads,
                            pack_vec,
                            pack_vec_exceed,
                            pack_zero
                        )                        
                    }
                }
            };
        }
        #[cfg(feature = "f16")]
        f16_prepack!(half::f16, crate::F16Vec, f32, crate::F32Vec);
        #[cfg(feature = "bf16")]
        f16_prepack!(half::bf16, crate::Bf16Vec, f32, crate::F32Vec);
        panic!("Unsupported type: {:?}", std::any::TypeId::of::<T>());
    } else {
        prepack_rhs_non_mp::<T>(n, m, k, rhs_ptr, rhs_strides, lhs_col_stride, num_threads)
    }
}
