use std::cmp::min;

use dyn_stack::DynStack;
use gemm_common::gemm::CACHELINE_ALIGN;

use crate::{
    Pointer, Zero,
    microkernel_trait::MatmulMicroKernel,
    utils::{
        L2_SLAB, L3_SLAB, PrePackedRhs, calculate_jobs, calculate_prgs, mt_intervals, pack_a,
        pack_a_mixed_precision, pack_b, pack_b_mixed_precision,
    },
    vec_size,
};

#[duplicate::duplicate_item(
    func_name       get_kernel                 get_horizontal_kernel                    extra_args;
    [matmul]        [get_kernel]               [get_horizontal_kernel]                  [];
    [matmul_post]   [get_kernel_with_post_op]  [get_horizontal_kernel_with_post_op]     [p + pb == k, i + ii, j + jj, _post_op.clone(), _post_op_vec.clone()];
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
    prepack_rhs: Option<PrePackedRhs>,
    _post_op: F1,
    _post_op_vec: F2,
) where
    T: MatmulMicroKernel + Send + Sync + Copy + Zero,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync,
    F2: Fn(<T as MatmulMicroKernel>::SelfVec, usize, usize) -> <T as MatmulMicroKernel>::SelfVec
        + Clone
        + Send
        + Sync,
{
    assert_eq!(nr % vec_size::<T>(), 0);

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;

    let packed_a = L3_SLAB.with(|mem| {
        if do_lhs_pack {
            let mut mem = mem.borrow_mut();
            let stack = DynStack::new(&mut mem);
            let (packed_a_storage, _) = stack.make_aligned_uninit::<T>(num_mr_blocks * mr * kc, 64);
            let packed_a = Pointer::new(
                packed_a_storage.as_mut_ptr() as *mut T,
                num_mr_blocks * mr * kc,
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

    let get_kernel = if mr == 1 {
        <T>::get_horizontal_kernel
    } else {
        <T>::get_kernel
    };

    let mut do_rhs_pack = true;
    if let Some(prepacked) = &prepack_rhs {
        assert_eq!(prepacked.kc, kc);
        assert_eq!(prepacked.mc, mc);
        assert_eq!(prepacked.nc, nc);
        assert_eq!(prepacked.mr, mr);
        assert_eq!(prepacked.nr, nr);
        assert_eq!(prepacked.num_threads, num_threads);
        assert_eq!(prepacked.prgs, prgs);
        assert_eq!(prepacked.rem_prgs, rem_prgs);
        assert_eq!(prepacked.intervals, intervals);
        assert_eq!(prepacked.rem_intervals, mc_rem_intervals);
        do_rhs_pack = false;
    }

    let panel_size = num_nr_blocks * nr * kc;

    for i in (0..m).step_by(mc) {
        let ib = min(mc, m - i);
        let prgs = if ib == mc { &prgs } else { &rem_prgs };
        let intervals = if ib == mc {
            &intervals
        } else {
            &mc_rem_intervals
        };
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let first_kiter = p == 0;
            let pb = min(kc, k - p);

            if do_lhs_pack {
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
                        num_mr_blocks,
                    );
                });
            }

            spindle::for_each_raw(num_threads, |tid| {
                let mut packed_b = if let Some(prepacked) = &prepack_rhs {
                    let buffer = if ib == mc {
                        &prepacked.buffers[p_idx][tid]
                    } else {
                        &prepacked.buffer_rems[p_idx][tid]
                    };
                    buffer.cast::<T>()
                } else {
                    L2_SLAB.with_borrow_mut(|mem| {
                        let stack = DynStack::new(mem);
                        let (mut packed_b_storage, _) =
                            stack.make_aligned_with::<T>(panel_size, CACHELINE_ALIGN, |_| T::ZERO);
                        Pointer::new(packed_b_storage.as_mut_ptr(), num_nr_blocks * nr * kc)
                    })
                };
                let packed_b_cpy = packed_b;

                let use_prg = prgs[tid];
                let use_start = intervals[tid].0;
                let use_end = intervals[tid].1;
                let j_start = use_prg[0] * nc;

                let mut job_count = use_start;
                let mut i_start = use_prg[1] * mr;
                let mut jj_start = use_prg[2] * nr;
                let need_full_pack = ib - i_start > mr;
                let mut panel_idx = 0;
                'outer: for j in (j_start..n).step_by(nc) {
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
                            need_full_pack,
                        );
                    } else {
                        packed_b = packed_b_cpy + panel_idx * (panel_size as i64);
                        panel_idx += 1;
                    }
                    let packed_a = if do_lhs_pack {
                        packed_a
                    } else {
                        a + ((i as i64) * lda + (p as i64) * lhs_col_stride)
                    };
                    for ii in (i_start..ib).step_by(mr) {
                        let mb = min(mr, ib - ii);
                        let micro_kernel = get_kernel(nr / vec_size::<T>(), mb);
                        for jj in (jj_start..jb).step_by(nr) {
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
                                    first_kiter,
                                    extra_args,
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
                                    first_kiter,
                                    extra_args,
                                );
                            }
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

pub(crate) fn single_thread_matmul<T>(
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
    prepack_rhs: Option<PrePackedRhs>,
) where
    T: MatmulMicroKernel + Send + Sync + Copy + Zero,
{
    assert_eq!(nr % vec_size::<T>(), 0);

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;

    let packed_a = L3_SLAB.with(|mem| {
        if do_lhs_pack {
            let mut mem = mem.borrow_mut();
            let stack = DynStack::new(&mut mem);
            let (packed_a_storage, _) = stack.make_aligned_uninit::<T>(num_mr_blocks * mr * kc, 64);
            let packed_a = Pointer::new(
                packed_a_storage.as_mut_ptr() as *mut T,
                num_mr_blocks * mr * kc,
            );
            packed_a
        } else {
            a
        }
    });

    let get_kernel = <T>::get_kernel;

    let mut do_rhs_pack = true;
    if let Some(prepacked) = &prepack_rhs {
        assert_eq!(prepacked.kc, kc);
        assert_eq!(prepacked.mc, mc);
        assert_eq!(prepacked.nc, nc);
        assert_eq!(prepacked.mr, mr);
        assert_eq!(prepacked.nr, nr);
        assert_eq!(prepacked.num_threads, num_threads);
        assert_eq!(prepacked.prgs, prgs);
        assert_eq!(prepacked.rem_prgs, rem_prgs);
        assert_eq!(prepacked.intervals, intervals);
        assert_eq!(prepacked.rem_intervals, mc_rem_intervals);
        do_rhs_pack = false;
    }

    let panel_size = num_nr_blocks * nr * kc;

    let mut packed_b_local_buffer = [0u8; 256 * 256 * 4];

    for i in (0..m).step_by(mc) {
        let ib = min(mc, m - i);
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let first_kiter = p == 0;
            let pb = min(kc, k - p);

            let packed_a = if do_lhs_pack {
                pack_a::<T>(
                    a + (i as i64) * lda + (p as i64) * lhs_col_stride,
                    packed_a,
                    lda,
                    lhs_col_stride,
                    ib,
                    pb,
                    kc,
                    mr,
                    0,
                    mb_per_thread,
                    num_mr_blocks,
                );
                packed_a
            } else {
                a + ((i as i64) * lda + (p as i64) * lhs_col_stride)
            };

            let mut packed_b = if let Some(prepacked) = &prepack_rhs {
                let buffer = if ib == mc {
                    &prepacked.buffers[p_idx][tid]
                } else {
                    &prepacked.buffer_rems[p_idx][tid]
                };
                buffer.cast::<T>()
            } else {
                L2_SLAB.with_borrow_mut(|mem| {
                    let stack = DynStack::new(mem);
                    let (mut packed_b_storage, _) =
                        stack.make_aligned_with::<T>(panel_size, CACHELINE_ALIGN, |_| T::ZERO);
                    Pointer::new(packed_b_storage.as_mut_ptr(), num_nr_blocks * nr * kc)
                })
            };
            let packed_b_cpy = packed_b;
            let mut panel_idx = 0;
            for j in (0..n).step_by(nc) {
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
                        0,
                        true,
                    );
                } else {
                    packed_b = packed_b_cpy + panel_idx * (panel_size as i64);
                    panel_idx += 1;
                }
                for ii in (0..ib).step_by(mr) {
                    let mb = min(mr, ib - ii);
                    let micro_kernel = get_kernel(nr / vec_size::<T>(), mb);
                    for jj in (0..jb).step_by(nr) {
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
                                first_kiter,
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
                                first_kiter,
                            );
                        }
                    }
                }
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
        usize,
    ),
    pack_vec_exceed: fn(*mut <T as MatmulMicroKernel>::MixedVec, usize),
    pack_zero: fn(&mut <T as MatmulMicroKernel>::MixedType, &T),
    vec_cast_back: fn(
        *mut <T as MatmulMicroKernel>::SelfVec,
        *const <T as MatmulMicroKernel>::MixedVec,
    ),
    cast_back: fn(&mut T, &<T as MatmulMicroKernel>::MixedType),
) where
    T: MatmulMicroKernel + Send + Sync + Copy + Zero,
    <T as MatmulMicroKernel>::MixedType: Send + Sync + Copy + Zero,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(<T as MatmulMicroKernel>::SelfVec, usize, usize) -> <T as MatmulMicroKernel>::SelfVec
        + Clone
        + Send
        + Sync
        + 'static,
{
    assert_eq!(nr % vec_size::<T>(), 0);

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;

    let packed_a = L3_SLAB.with(|mem| {
        let mut mem = mem.borrow_mut();
        let stack = DynStack::new(&mut mem);
        let (packed_a_storage, _) = stack
            .make_aligned_uninit::<<T as MatmulMicroKernel>::MixedType>(
                num_mr_blocks * mr * kc,
                64,
            );
        let packed_a = Pointer::new(
            packed_a_storage.as_mut_ptr() as *mut <T as MatmulMicroKernel>::MixedType,
            num_mr_blocks * mr * kc,
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
        assert_eq!(prepacked.mc, mc);
        assert_eq!(prepacked.nc, nc);
        assert_eq!(prepacked.mr, mr);
        assert_eq!(prepacked.nr, nr);
        assert_eq!(prepacked.num_threads, num_threads);
        assert_eq!(prepacked.prgs, prgs);
        assert_eq!(prepacked.rem_prgs, rem_prgs);
        assert_eq!(prepacked.intervals, intervals);
        assert_eq!(prepacked.rem_intervals, mc_rem_intervals);
        do_rhs_pack = false;
    }

    let panel_size = num_nr_blocks * nr * kc;

    for i in (0..m).step_by(mc) {
        let ib = min(mc, m - i);
        let prgs = if ib == mc { &prgs } else { &rem_prgs };
        let intervals = if ib == mc {
            &intervals
        } else {
            &mc_rem_intervals
        };
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
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
                    pack_zero,
                );
            });
            spindle::for_each_raw(num_threads, |tid| {
                let mut packed_b = if let Some(prepacked) = &prepack_rhs {
                    let buffer = if ib == mc {
                        &prepacked.buffers[p_idx][tid]
                    } else {
                        &prepacked.buffer_rems[p_idx][tid]
                    };
                    buffer.cast::<<T as MatmulMicroKernel>::MixedType>()
                } else {
                    L2_SLAB.with(|mem| {
                        let mut mem = mem.borrow_mut();
                        let stack = DynStack::new(&mut mem);
                        let (packed_b_storage, _) = stack
                            .make_aligned_uninit::<<T as MatmulMicroKernel>::MixedType>(
                                num_nr_blocks * nr * kc,
                                64,
                            );
                        Pointer::new(
                            packed_b_storage.as_mut_ptr()
                                as *mut <T as MatmulMicroKernel>::MixedType,
                            num_nr_blocks * nr * kc,
                        )
                    })
                };
                let packed_b_cpy = packed_b;
                let use_prg = prgs[tid];
                let use_start = intervals[tid].0;
                let use_end = intervals[tid].1;
                let j_start = use_prg[0] * nc;

                let mut job_count = use_start;
                let mut i_start = use_prg[1] * mr;
                let mut jj_start = use_prg[2] * nr;
                let mut panel_idx = 0;
                'outer: for j in (j_start..n).step_by(nc) {
                    let jb = min(nc, n - j);
                    let c = out.clone() + (i as i64) * ldc + (j as i64);
                    if do_rhs_pack {
                        pack_b_mixed_precision::<
                            T,
                            <T as MatmulMicroKernel>::MixedType,
                            <T as MatmulMicroKernel>::SelfVec,
                            <T as MatmulMicroKernel>::MixedVec,
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
                            pack_zero,
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
                                args,
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
