use std::cmp::min;

use dyn_stack::DynStack;
use gemm_common::gemm::CACHELINE_ALIGN;
use rayon::iter::{ IntoParallelIterator, ParallelIterator };

use crate::{
    ALIGN,
    Pointer,
    Zero,
    microkernel_trait::MatmulMicroKernel,
    utils::{
        L2_SLAB,
        L3_SLAB,
        PrePackedRhs,
        calculate_jobs,
        calculate_prgs,
        mt_intervals,
        pack_a,
        pack_a_mixed_precision,
        pack_b,
        pack_b_mixed_precision,
    },
    vec_size,
};

#[duplicate::duplicate_item(
    func_name       mv_method       get_kernel                 extra_args;
    [matmul]        [mv]            [get_kernel]               [first_kiter];
    [matmul_post]   [mv_post]       [get_kernel_with_post_op]  [first_kiter,p + pb == k, i + ii, j + jj, post_op.clone(), post_op_vec.clone()];
)]
pub(crate) fn func_name<T, F1, F2>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    kc: usize,
    mc: usize,
    nc: usize,
    nr: usize,
    mr: usize,
    do_lhs_pack: bool,
    mut num_threads: usize,
    prepack_rhs: Option<&PrePackedRhs>,
    #[allow(unused_variables)] post_op: F1,
    #[allow(unused_variables)] post_op_vec: F2
)
    where
        T: MatmulMicroKernel + Send + Sync + Copy + Zero,
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
    assert_eq!(nr % vec_size::<T>(), 0);

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;

    let packed_a = L3_SLAB.with(|mem| {
        if do_lhs_pack {
            let mut mem = mem.borrow_mut();
            let stack = DynStack::new(&mut mem);
            let (packed_a_storage, _) = stack.make_aligned_uninit::<T>(
                num_mr_blocks * mr * kc,
                ALIGN
            );
            let packed_a = Pointer::new(
                packed_a_storage.as_mut_ptr() as *mut T,
                (num_mr_blocks * mr * kc) as i64
            );
            packed_a
        } else {
            a
        }
    });

    let mc_jobs = calculate_jobs(n, nc, mr, nr, mc);
    let mc_rem_jobs = calculate_jobs(n, nc, mr, nr, m % mc);
    num_threads = num_threads.min(mc_jobs);
    let mb_per_thread = num_mr_blocks.div_ceil(num_threads);
    let intervals = mt_intervals(mc_jobs, num_threads);
    let mc_rem_intervals = mt_intervals(mc_rem_jobs, num_threads);
    let prgs = calculate_prgs(n, nc, mr, nr, mc, &intervals);
    let rem_prgs = calculate_prgs(n, nc, mr, nr, m % mc, &mc_rem_intervals);

    let get_kernel = <T>::get_kernel;

    let mut do_rhs_pack = true;
    if let Some(prepacked) = &prepack_rhs {
        assert_eq!(prepacked.kc, kc);
        assert_eq!(prepacked.nr, nr);
        do_rhs_pack = false;
    }

    let panel_size = num_nr_blocks * nr * kc;

    let mut i = 0;
    while i < m {
        let ib = min(mc, m - i);
        let prgs = if ib == mc { &prgs } else { &rem_prgs };
        let intervals = if ib == mc { &intervals } else { &mc_rem_intervals };

        let mut p = 0;
        while p < k {
            let first_kiter = p == 0;
            let pb = min(kc, k - p);

            let packed_a = if do_lhs_pack {
                spindle::for_each_raw(num_threads, |tid| {
                    pack_a::<T>(
                        a + (i as i64) * lda + (p as i64) * lhs_col_stride,
                        packed_a,
                        lda,
                        lhs_col_stride,
                        ib,
                        pb,
                        kc,
                        mr,
                        tid,
                        mb_per_thread,
                        num_mr_blocks
                    );
                });
                packed_a
            } else {
                a + ((i as i64) * lda + (p as i64) * lhs_col_stride)
            };
            spindle::for_each_raw(num_threads, |tid| {
                let use_prg = prgs[tid];
                let use_start = intervals[tid].0;
                let use_end = intervals[tid].1;
                let j_idx = use_prg[0];
                let mut j = j_idx * nc;

                let mut job_count = use_start;
                let mut i_start = use_prg[1] * mr;
                let mut jj_start = use_prg[2] * nr;
                let need_full_pack = ib - i_start > mr;

                let mut packed_b = if let Some(prepacked) = &prepack_rhs {
                    let buffer = prepacked.buffer.0.cast::<T>() + ((j_idx * panel_size) as i64);
                    buffer
                } else {
                    L2_SLAB.with_borrow_mut(|mem| {
                        let stack = DynStack::new(mem);
                        let (packed_b_storage, _) = stack.make_aligned_uninit::<T>(
                            panel_size,
                            CACHELINE_ALIGN
                        );
                        Pointer::new(
                            packed_b_storage.as_mut_ptr() as *mut T,
                            (num_nr_blocks * nr * kc) as i64
                        )
                    })
                };
                let packed_b_cpy = packed_b;

                let mut j_idx = 0;
                'outer: while j < n {
                    let jb = min(nc, n - j);
                    let c = out + (i as i64) * ldc + (j as i64);
                    if do_rhs_pack {
                        pack_b::<T, <T as MatmulMicroKernel>::SelfVec>(
                            b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                            packed_b,
                            ldb,
                            rhs_col_stride,
                            jb,
                            pb,
                            kc,
                            nr,
                            jj_start,
                            need_full_pack
                        );
                    } else {
                        packed_b = packed_b_cpy + ((j_idx * panel_size) as i64);
                    }

                    let mut ii = i_start;
                    while ii < ib {
                        let mb = min(mr, ib - ii);
                        let micro_kernel = get_kernel(nr / vec_size::<T>(), mb);

                        let mut jj = jj_start;
                        while jj < jb {
                            let jjb = min(nr, jb - jj);
                            let packed_b = packed_b + (jj as i64) * (kc as i64);
                            if do_lhs_pack {
                                micro_kernel(
                                    packed_a + (kc as i64) * (ii as i64),
                                    packed_b,
                                    c + (ii as i64) * ldc + (jj as i64),
                                    ldc,
                                    1,
                                    pb,
                                    jjb,
                                    mb as i64,
                                    extra_args
                                );
                            } else {
                                micro_kernel(
                                    packed_a + (ii as i64) * lda,
                                    packed_b,
                                    c + (ii as i64) * ldc + (jj as i64),
                                    ldc,
                                    lda,
                                    pb,
                                    jjb,
                                    lhs_col_stride,
                                    extra_args
                                );
                            }
                            job_count += 1;
                            if job_count >= use_end {
                                break 'outer;
                            }
                            jj += nr;
                        }
                        jj_start = 0;

                        ii += mr;
                    }
                    i_start = 0;

                    j += nc;
                    j_idx += 1;
                }
            });
            p += kc;
        }
        i += mc;
    }
}

/// return the bytes size of the prepacked needed
pub(crate) fn prepack_b_size<T>(n: usize, k: usize, kc: usize, nr: usize) -> usize {
    assert_eq!(nr % vec_size::<T>(), 0);
    let panel_size = nr * kc;
    let num_kc = k.div_ceil(kc);
    let num_nr = n.div_ceil(nr);
    return num_kc * num_nr * panel_size * vec_size::<T>() * std::mem::size_of::<T>();
}

pub(crate) fn prepack_b<T>(
    b: Pointer<T>,
    buffer: Pointer<T>,
    n: usize,
    k: usize,
    rhs_strides: [i64; 2],
    kc: usize,
    nc: usize,
    nr: usize,
    parallel: bool
)
    where T: MatmulMicroKernel + Send + Sync + Copy + Zero
{
    assert_eq!(nr % vec_size::<T>(), 0);

    let panel_size = nr * kc;
    let num_nr = n.div_ceil(nr);
    let ldb = rhs_strides[0];
    let rhs_col_stride = rhs_strides[1];
    if parallel {
        let mut work_items = Vec::new();
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let pb = min(kc, k - p);
            let buffer = buffer + ((p_idx * num_nr * panel_size) as i64);
            for (j_idx, j) in (0..n).step_by(nc).enumerate() {
                let jb = min(nc, n - j);
                work_items.push((
                    jb,
                    pb,
                    b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                    buffer + ((j_idx * panel_size * (nc / nr)) as i64),
                ));
            }
        }
        work_items.into_par_iter().for_each(|(jb, pb, b, buffer)| {
            pack_b::<T, <T as MatmulMicroKernel>::SelfVec>(
                b,
                buffer,
                ldb,
                rhs_col_stride,
                jb,
                pb,
                kc,
                nr,
                0,
                true
            );
        });
    } else {
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let pb = min(kc, k - p);
            let buffer = buffer + ((p_idx * num_nr * panel_size) as i64);
            for (j_idx, j) in (0..n).step_by(nc).enumerate() {
                let jb = min(nc, n - j);
                pack_b::<T, <T as MatmulMicroKernel>::SelfVec>(
                    b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                    buffer + ((j_idx * panel_size * (nc / nr)) as i64),
                    ldb,
                    rhs_col_stride,
                    jb,
                    pb,
                    kc,
                    nr,
                    0,
                    true
                );
            }
        }
    }
}

#[duplicate::duplicate_item(
    func_name           ty      get_kernel                                  args;
    [matmul_mp]         [T]     [get_mixed_precision_kernel]                [vec_cast_back, cast_back];
    [matmul_mp_post]    [T]     [get_mixed_precision_kernel_with_post_op]   [p + pb == k, i + ii, j + jj, vec_cast_back, cast_back, _post_op.clone(), _post_op_vec.clone()];
)]
pub(crate) fn func_name<T, F1, F2>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    kc: usize,
    mc: usize,
    nc: usize,
    nr: usize,
    mr: usize,
    mut num_threads: usize,
    prepack_rhs: Option<PrePackedRhs>,
    _post_op: F1,
    _post_op_vec: F2,
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
        T: MatmulMicroKernel + Send + Sync + Copy + Zero,
        <T as MatmulMicroKernel>::MixedType: Send + Sync + Copy + Zero,
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
    assert_eq!(nr % vec_size::<T>(), 0);

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;

    let packed_a = L3_SLAB.with(|mem| {
        let mut mem = mem.borrow_mut();
        let stack = DynStack::new(&mut mem);
        let (packed_a_storage, _) =
            stack.make_aligned_uninit::<<T as MatmulMicroKernel>::MixedType>(
                num_mr_blocks * mr * kc,
                ALIGN
            );
        let packed_a = Pointer::new(
            packed_a_storage.as_mut_ptr() as *mut <T as MatmulMicroKernel>::MixedType,
            (num_mr_blocks * mr * kc) as i64
        );
        packed_a
    });

    let mc_jobs = calculate_jobs(n, nc, mr, nr, mc);
    let mc_rem_jobs = calculate_jobs(n, nc, mr, nr, m % mc);
    num_threads = num_threads.min(mc_jobs);
    let mb_per_thread = num_mr_blocks.div_ceil(num_threads);
    let intervals = mt_intervals(mc_jobs, num_threads);
    let mc_rem_intervals = mt_intervals(mc_rem_jobs, num_threads);
    let prgs = calculate_prgs(n, nc, mr, nr, mc, &intervals);
    let rem_prgs = calculate_prgs(n, nc, mr, nr, m % mc, &mc_rem_intervals);

    let mut do_rhs_pack = true;
    if let Some(prepacked) = &prepack_rhs {
        assert_eq!(prepacked.kc, kc);
        assert_eq!(prepacked.nc, nc);
        assert_eq!(prepacked.nr, nr);
        do_rhs_pack = false;
    }

    let panel_size = num_nr_blocks * nr * kc;

    for i in (0..m).step_by(mc) {
        let ib = min(mc, m - i);
        let prgs = if ib == mc { &prgs } else { &rem_prgs };
        let intervals = if ib == mc { &intervals } else { &mc_rem_intervals };
        for p in (0..k).step_by(kc) {
            let first_kiter = p == 0;
            let pb = min(kc, k - p);
            spindle::for_each_raw(num_threads, |tid| {
                pack_a_mixed_precision::<T, <T as MatmulMicroKernel>::MixedType>(
                    a + (i as i64) * lda + (p as i64) * lhs_col_stride,
                    packed_a,
                    lda,
                    lhs_col_stride,
                    ib,
                    pb,
                    kc,
                    mr,
                    tid,
                    mb_per_thread,
                    num_mr_blocks,
                    pack_zero
                );
            });
            spindle::for_each_raw(num_threads, |tid| {
                let use_prg = prgs[tid];
                let use_start = intervals[tid].0;
                let use_end = intervals[tid].1;
                let j_idx = use_prg[0];
                let j_start = j_idx * nc;

                let mut job_count = use_start;
                let mut i_start = use_prg[1] * mr;
                let mut jj_start = use_prg[2] * nr;
                let mut packed_b = if let Some(prepacked) = &prepack_rhs {
                    let buffer =
                        prepacked.buffer.0 +
                        (
                            (j_idx *
                                panel_size *
                                std::mem::size_of::<<T as MatmulMicroKernel>::MixedType>()) as i64
                        );
                    buffer.cast::<<T as MatmulMicroKernel>::MixedType>()
                } else {
                    L2_SLAB.with(|mem| {
                        let mut mem = mem.borrow_mut();
                        let stack = DynStack::new(&mut mem);
                        let (packed_b_storage, _) =
                            stack.make_aligned_uninit::<<T as MatmulMicroKernel>::MixedType>(
                                num_nr_blocks * nr * kc,
                                ALIGN
                            );
                        Pointer::new(
                            packed_b_storage.as_mut_ptr() as *mut <T as MatmulMicroKernel>::MixedType,
                            (num_nr_blocks * nr * kc) as i64
                        )
                    })
                };
                let mut panel_idx = 0;
                let packed_b_cpy = packed_b;
                'outer: for j in (j_start..n).step_by(nc) {
                    let jb = min(nc, n - j);
                    let c = out.clone() + (i as i64) * ldc + (j as i64);
                    if do_rhs_pack {
                        pack_b_mixed_precision::<
                            T,
                            <T as MatmulMicroKernel>::MixedType,
                            <T as MatmulMicroKernel>::SelfVec,
                            <T as MatmulMicroKernel>::MixedVec
                        >(
                            b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                            packed_b,
                            ldb,
                            rhs_col_stride,
                            jb,
                            pb,
                            kc,
                            nr,
                            pack_vec,
                            pack_vec_exceed,
                            pack_zero
                        );
                    } else {
                        packed_b = packed_b_cpy + panel_idx * (panel_size as i64);
                        panel_idx += 1;
                    }
                    let packed_a = packed_a.clone();
                    for ii in (i_start..ib).step_by(mr) {
                        let mb = min(mr, ib - ii);
                        let micro_kernel = <T>::get_kernel(nr / vec_size::<T>(), mb);

                        for jj in (jj_start..jb).step_by(nr) {
                            let jjb = min(nr, jb - jj);
                            micro_kernel(
                                packed_a + (kc as i64) * (ii as i64),
                                packed_b + (jj as i64) * (kc as i64),
                                c + (ii as i64) * ldc + (jj as i64),
                                ldc,
                                1,
                                kc,
                                jjb,
                                mb as i64,
                                first_kiter,
                                args
                            );
                            job_count += 1;
                            if job_count >= use_end {
                                break 'outer;
                            }
                        }
                        jj_start = 0;
                    }
                    i_start = 0;
                }
            });
        }
    }
}

// #[duplicate::duplicate_item(
//     func_name   get_kernel                      extra_args;
//     [mv]        [get_gemv_kernel]               [];
//     [mv_post]   [get_gemv_kernel_with_post_op]  [m_offset, n_offset, post_op, post_op_vec];
// )]
// pub(crate) fn func_name<T, F, F2>(
//     a: Pointer<T>,
//     b: Pointer<T>,
//     out: Pointer<T>,
//     #[allow(unused_variables)] m_offset: usize,
//     #[allow(unused_variables)] n_offset: usize,
//     n: usize,
//     k: usize,
//     ldb: i64,
//     lhs_col_stride: i64,
//     kc: usize,
//     nc: usize,
//     nr: usize,
//     prepack_rhs: Option<&PrePackedRhs>,
//     #[allow(unused_variables)] post_op: F,
//     #[allow(unused_variables)] post_op_vec: F2
// )
//     where
//         T: MatmulMicroKernel + Send + Sync + Copy + Zero,
//         F: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
//         F2: Fn(
//             <T as MatmulMicroKernel>::SelfVec,
//             usize,
//             usize
//         ) -> <T as MatmulMicroKernel>::SelfVec +
//             Clone +
//             Send +
//             Sync +
//             'static
// {
//     let micro_kernel = <T>::get_kernel();

//     if let Some(prepacked_rhs) = &prepack_rhs {
//         assert_eq!(prepacked_rhs.kc, kc);
//         assert_eq!(prepacked_rhs.nc, nc);
//         assert_eq!(prepacked_rhs.nr, nr);
//         assert_eq!(prepacked_rhs.buffers.len(), 1);
//         let prepacked_b = prepacked_rhs.buffers[0].cast::<T>();
//         micro_kernel(a, prepacked_b, out, n, k, ldb, lhs_col_stride, extra_args)
//     } else {
//         micro_kernel(a, b, out, n, k, ldb, lhs_col_stride, extra_args)
//     }
// }

// #[duplicate::duplicate_item(
//     func_name       get_kernel                      extra_args;
//     [mv_mp]         [get_gemv_kernel_mp]               [vec_cast_back, cast_back];
//     [mv_post_mp]    [get_gemv_kernel_mp_with_post_op]  [m_offset, n_offset, vec_cast_back, cast_back, post_op, post_op_vec];
// )]
// pub(crate) fn func_name<T, F, F2>(
//     a: Pointer<<T as MatmulMicroKernel>::MixedType>,
//     b: Pointer<<T as MatmulMicroKernel>::MixedType>,
//     out: Pointer<T>,
//     #[allow(unused_variables)] m_offset: usize,
//     #[allow(unused_variables)] n_offset: usize,
//     n: usize,
//     k: usize,
//     ldb: i64,
//     lhs_col_stride: i64,
//     kc: usize,
//     nc: usize,
//     nr: usize,
//     prepack_rhs: Option<&PrePackedRhs>,
//     #[allow(unused_variables)] post_op: F,
//     #[allow(unused_variables)] post_op_vec: F2,
//     vec_cast_back: fn(
//         *mut <T as MatmulMicroKernel>::SelfVec,
//         *const <T as MatmulMicroKernel>::MixedVec
//     ),
//     cast_back: fn(&mut T, &<T as MatmulMicroKernel>::MixedType)
// )
//     where
//         T: MatmulMicroKernel + Copy + Zero,
//         F: Fn(T, usize, usize) -> T + Clone + 'static,
//         F2: Fn(
//             <T as MatmulMicroKernel>::SelfVec,
//             usize,
//             usize
//         ) -> <T as MatmulMicroKernel>::SelfVec +
//             Clone +
//             'static
// {
//     let micro_kernel = <T>::get_kernel();

//     if let Some(prepacked_rhs) = &prepack_rhs {
//         assert_eq!(prepacked_rhs.kc, kc);
//         assert_eq!(prepacked_rhs.nc, nc);
//         assert_eq!(prepacked_rhs.nr, nr);
//         assert_eq!(prepacked_rhs.buffers.len(), 1);
//         let prepacked_b = prepacked_rhs.buffers[0].cast::<<T as MatmulMicroKernel>::MixedType>();
//         micro_kernel(a, prepacked_b, out, n, k, nc as i64, lhs_col_stride, extra_args)
//     } else {
//         micro_kernel(a, b, out, n, k, ldb, lhs_col_stride, extra_args)
//     }
// }
