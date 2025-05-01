use std::cmp::min;

use dyn_stack::DynStack;
use hpt_common::{ Pointer, shape::shape_utils::mt_intervals };
use hpt_traits::tensor::CommonBounds;

use super::microkernel_trait::MatmulMicroKernel;
use crate::ops::tensor::matmul::common::{ L2_SLAB, L3_SLAB, calculate_jobs, calculate_prgs };
use crate::ALIGN;
use hpt_types::traits::VecTrait;

macro_rules! call_microkernel {
    (
        true,
        $packed_a:expr,
        $packed_b:expr,
        $c:expr,
        $micro_kernel:expr,
        $ldc:expr,
        $pb:expr,
        $jjb:expr,
        $mb:expr,
        $first_kiter:expr,
        $m_idx:expr,
        $n_idx:expr,
        $last_k_iter:expr,
        $post_op:expr,
        $post_op_vec:expr
    ) => {
        $micro_kernel(
            $packed_a,
            $packed_b,
            $c,
            $ldc,
            1,
            $pb,
            $jjb,
            $mb as i64,
            $first_kiter,
            $last_k_iter,
            $m_idx,
            $n_idx,
            $post_op.clone(),
            $post_op_vec.clone(),
        );
    };
    (
        false,
        $packed_a:expr,
        $packed_b:expr,
        $c:expr,
        $micro_kernel:expr,
        $ldc:expr,
        $pb:expr,
        $jjb:expr,
        $mb:expr,
        $first_kiter:expr,
        $m_idx:expr,
        $n_idx:expr,
        $last_k_iter:expr,
        $post_op:expr,
        $post_op_vec:expr
    ) => {
        $micro_kernel(
            $packed_a,
            $packed_b,
            $c,
            $ldc,
            1,
            $pb,
            $jjb,
            $mb as i64,
            $first_kiter,
        );
    };
}

macro_rules! call_non_packed_microkernel {
    (
        true,
        $packed_a:expr,
        $packed_b:expr,
        $c:expr,
        $micro_kernel:expr,
        $ldc:expr,
        $lda:expr,
        $pb:expr,
        $jjb:expr,
        $lhs_col_stride:expr,
        $first_kiter:expr,
        $m_idx:expr,
        $n_idx:expr,
        $last_k_iter:expr,
        $post_op:expr,
        $post_op_vec:expr
    ) => {
        $micro_kernel(
            $packed_a,
            $packed_b,
            $c,
            $ldc,
            $lda,
            $pb,
            $jjb,
            $lhs_col_stride,
            $first_kiter,
            $last_k_iter,
            $m_idx,
            $n_idx,
            $post_op.clone(),
            $post_op_vec.clone(),
        );
    };
    (
        false,
        $packed_a:expr,
        $packed_b:expr,
        $c:expr,
        $micro_kernel:expr,
        $ldc:expr,
        $lda:expr,
        $pb:expr,
        $jjb:expr,
        $lhs_col_stride:expr,
        $first_kiter:expr,
        $m_idx:expr,
        $n_idx:expr,
        $last_k_iter:expr,
        $post_op:expr,
        $post_op_vec:expr
    ) => {
        $micro_kernel(
            $packed_a,
            $packed_b,
            $c,
            $ldc,
            $lda,
            $pb,
            $jjb,
            $lhs_col_stride,
            $first_kiter,
        );
    };
}

#[duplicate::duplicate_item(
    func_name ty  get_kernel  need_post_op;
    [matmul]  [T] [get_kernel] [false];
    [matmul_post]  [T] [get_kernel_with_post_op] [true];
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
    with_locked: bool,
    _post_op: F1,
    _post_op_vec: F2
)
    where
        T: CommonBounds + MatmulMicroKernel,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static
{
    assert_eq!(
        nr % T::Vec::SIZE,
        0,
        "nr must be a multiple of {} for type {}",
        T::Vec::SIZE,
        T::STR
    );

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
            a.clone()
        }
    });

    let mc_jobs = calculate_jobs(n, nc, mr, nr, mc);
    let mc_rem_jobs = calculate_jobs(n, nc, mr, nr, m % mc);
    num_threads = num_threads.min(mc_jobs);
    let jobs_per_thread =
        mc_jobs.div_ceil(num_threads) * mc + ((mc_rem_jobs.div_ceil(num_threads) * m) % mc);
    if jobs_per_thread < 1024 {
        while num_threads > 1 {
            num_threads -= 1;
            let jobs_per_thread =
                mc_jobs.div_ceil(num_threads) * mc + ((mc_rem_jobs.div_ceil(num_threads) * m) % mc);
            if jobs_per_thread >= 1024 {
                break;
            }
        }
    }
    let mb_per_thread = num_mr_blocks.div_ceil(num_threads);
    let intervals = mt_intervals(mc_jobs, num_threads);
    let mc_rem_intervals = mt_intervals(mc_rem_jobs, num_threads);
    let prgs = calculate_prgs(n, nc, mr, nr, mc, &intervals);
    let rem_prgs = calculate_prgs(n, nc, mr, nr, m % mc, &mc_rem_intervals);

    // if num_threads == 1 {
    //     for i in (0..m).step_by(mc) {
    //         let ib = min(mc, m - i);
    //         for p in (0..k).step_by(kc) {
    //             let first_kiter = p == 0;
    //             let pb = min(kc, k - p);
    //             if do_lhs_pack {
    //                 pack_a::<T>(
    //                     a + (i as i64) * lda + (p as i64) * lhs_col_stride,
    //                     packed_a,
    //                     lda,
    //                     lhs_col_stride,
    //                     ib,
    //                     pb,
    //                     kc,
    //                     mr,
    //                     0,
    //                     mb_per_thread,
    //                     num_mr_blocks
    //                 );
    //             }

    //             L2_SLAB.with_borrow_mut(|mem| {
    //                 let stack = DynStack::new(mem);
    //                 let (mut packed_b_storage, _) = stack.make_aligned_with::<T>(
    //                     num_nr_blocks * nr * kc,
    //                     ALIGN,
    //                     |_| T::ZERO
    //                 );
    //                 let packed_b = Pointer::new(
    //                     packed_b_storage.as_mut_ptr() as *mut T,
    //                     (num_nr_blocks * nr * kc) as i64
    //                 );
    //                 let use_prg = if ib == mc { prgs[0] } else { rem_prgs[0] };
    //                 let use_start = if ib == mc { intervals[0].0 } else { mc_rem_intervals[0].0 };
    //                 let use_end = if ib == mc { intervals[0].1 } else { mc_rem_intervals[0].1 };
    //                 let j_start = use_prg[0] * nc;
    //                 let mut job_count = use_start;
    //                 let mut i_start = use_prg[1] * mr;
    //                 let mut jj_start = use_prg[2] * nr;
    //                 let need_full_pack = ib - i_start > mr;
    //                 'outer: for j in (j_start..n).step_by(nc) {
    //                     let jb = min(nc, n - j);
    //                     let c = out + (i as i64) * ldc + (j as i64);
    //                     pack_b::<T>(
    //                         b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
    //                         packed_b,
    //                         ldb,
    //                         rhs_col_stride,
    //                         jb,
    //                         pb,
    //                         kc,
    //                         nr,
    //                         jj_start,
    //                         need_full_pack
    //                     );
    //                     let packed_a = if do_lhs_pack {
    //                         packed_a
    //                     } else {
    //                         a + ((i as i64) * lda + (p as i64) * lhs_col_stride)
    //                     };
    //                     for ii in (i_start..ib).step_by(mr) {
    //                         let mb = min(mr, ib - ii);
    //                         let micro_kernel = <T>::get_kernel(nr / <T>::Vec::SIZE, mb);
    //                         for jj in (jj_start..jb).step_by(nr) {
    //                             let jjb = min(nr, jb - jj);
    //                             let packed_b = packed_b + (jj as i64) * (kc as i64);
    //                             if do_lhs_pack {
    //                                 call_microkernel!(
    //                                     need_post_op,
    //                                     packed_a + (kc as i64) * (ii as i64),
    //                                     packed_b,
    //                                     c + (ii as i64) * ldc + (jj as i64),
    //                                     micro_kernel,
    //                                     ldc,
    //                                     pb,
    //                                     jjb,
    //                                     mb as i64,
    //                                     first_kiter,
    //                                     i + ii,
    //                                     j + jj,
    //                                     p + pb == k,
    //                                     _post_op,
    //                                     _post_op_vec
    //                                 );
    //                             } else {
    //                                 call_non_packed_microkernel!(
    //                                     need_post_op,
    //                                     packed_a + (ii as i64) * lda,
    //                                     packed_b,
    //                                     c + (ii as i64) * ldc + (jj as i64),
    //                                     micro_kernel,
    //                                     ldc,
    //                                     lda,
    //                                     pb,
    //                                     jjb,
    //                                     lhs_col_stride,
    //                                     first_kiter,
    //                                     i + ii,
    //                                     j + jj,
    //                                     p + pb == k,
    //                                     _post_op,
    //                                     _post_op_vec
    //                                 );
    //                             }
    //                             job_count += 1;
    //                             if job_count >= use_end {
    //                                 break 'outer;
    //                             }
    //                         }
    //                         jj_start = 0;
    //                     }
    //                     i_start = 0;
    //                 }
    //             });
    //         }
    //     }
    // } else {
        let func = || {
            for i in (0..m).step_by(mc) {
                let ib = min(mc, m - i);
                for p in (0..k).step_by(kc) {
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
                                num_mr_blocks
                            );
                        });
                    }

                    spindle::for_each_raw(num_threads, |tid| {
                        L2_SLAB.with_borrow_mut(|mem| {
                            let stack = DynStack::new(mem);
                            let (mut packed_b_storage, _) = stack.make_aligned_with::<T>(
                                num_nr_blocks * nr * kc,
                                ALIGN,
                                |_| T::ZERO
                            );
                            let packed_b = Pointer::new(
                                packed_b_storage.as_mut_ptr() as *mut T,
                                (num_nr_blocks * nr * kc) as i64
                            );
                            let use_prg = if ib == mc { prgs[tid] } else { rem_prgs[tid] };
                            let use_start = if ib == mc {
                                intervals[tid].0
                            } else {
                                mc_rem_intervals[tid].0
                            };
                            let use_end = if ib == mc {
                                intervals[tid].1
                            } else {
                                mc_rem_intervals[tid].1
                            };
                            let j_start = use_prg[0] * nc;
                            let mut job_count = use_start;
                            let mut i_start = use_prg[1] * mr;
                            let mut jj_start = use_prg[2] * nr;
                            let need_full_pack = ib - i_start > mr;
                            'outer: for j in (j_start..n).step_by(nc) {
                                let jb = min(nc, n - j);
                                let c = out + (i as i64) * ldc + (j as i64);
                                pack_b::<T>(
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
                                let packed_a = if do_lhs_pack {
                                    packed_a
                                } else {
                                    a + ((i as i64) * lda + (p as i64) * lhs_col_stride)
                                };
                                for ii in (i_start..ib).step_by(mr) {
                                    let mb = min(mr, ib - ii);
                                    let micro_kernel = <T>::get_kernel(nr / <T>::Vec::SIZE, mb);
                                    for jj in (jj_start..jb).step_by(nr) {
                                        let jjb = min(nr, jb - jj);
                                        let packed_b = packed_b + (jj as i64) * (kc as i64);
                                        if do_lhs_pack {
                                            call_microkernel!(
                                                need_post_op,
                                                packed_a + (kc as i64) * (ii as i64),
                                                packed_b,
                                                c + (ii as i64) * ldc + (jj as i64),
                                                micro_kernel,
                                                ldc,
                                                pb,
                                                jjb,
                                                mb as i64,
                                                first_kiter,
                                                i + ii,
                                                j + jj,
                                                p + pb == k,
                                                _post_op,
                                                _post_op_vec
                                            );
                                        } else {
                                            call_non_packed_microkernel!(
                                                need_post_op,
                                                packed_a + (ii as i64) * lda,
                                                packed_b,
                                                c + (ii as i64) * ldc + (jj as i64),
                                                micro_kernel,
                                                ldc,
                                                lda,
                                                pb,
                                                jjb,
                                                lhs_col_stride,
                                                first_kiter,
                                                i + ii,
                                                j + jj,
                                                p + pb == k,
                                                _post_op,
                                                _post_op_vec
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
                    });
                }
            }
        };
        if with_locked {
            func();
        } else {
            spindle::with_lock(num_threads, func);
        }
    // }

    // THREAD_POOL.with_borrow(|pool| {
    //     pool.parallel_for(
    //         (0..num_threads)
    //             .into_iter()
    //             .zip(prgs.into_iter())
    //             .zip(rem_prgs.into_iter())
    //             .zip(intervals.into_iter())
    //             .zip(mc_rem_intervals.into_iter()),
    //         move |((((tid, prg), rem_prg), (start, end)), (start_rem, end_rem)), _| {
    //             L2_SLAB.with_borrow_mut(|mem| {
    //                 let stack = DynStack::new(mem);
    //                 let (mut packed_b_storage, _) =
    //                     stack.make_aligned_with::<T>(num_nr_blocks * nr * kc, ALIGN, |_| T::ZERO);
    //                 let packed_b = Pointer::new(
    //                     packed_b_storage.as_mut_ptr() as *mut T,
    //                     (num_nr_blocks * nr * kc) as i64,
    //                 );
    //                 for i in (0..m).step_by(mc) {
    //                     let ib = min(mc, m - i);
    //                     let use_prg = if ib == mc { prg } else { rem_prg };
    //                     let use_start = if ib == mc { start } else { start_rem };
    //                     let use_end = if ib == mc { end } else { end_rem };
    //                     let j_start = use_prg[0] * nc;
    //                     for p in (0..k).step_by(kc) {
    //                         let first_kiter = p == 0;
    //                         let pb = min(kc, k - p);
    //                         if do_lhs_pack {
    //                             pack_a::<T>(
    //                                 a + (i as i64) * lda + (p as i64) * lhs_col_stride,
    //                                 packed_a,
    //                                 lda,
    //                                 lhs_col_stride,
    //                                 ib,
    //                                 pb,
    //                                 kc,
    //                                 mr,
    //                                 tid,
    //                                 mb_per_thread,
    //                                 num_mr_blocks,
    //                             );
    //                             barrier.wait();
    //                         }

    //                         let mut job_count = use_start;
    //                         let mut i_start = use_prg[1] * mr;
    //                         let mut jj_start = use_prg[2] * nr;
    //                         let need_full_pack = ib - i_start > mr;
    //                         'outer: for j in (j_start..n).step_by(nc) {
    //                             let jb = min(nc, n - j);
    //                             let c = out + (i as i64) * ldc + (j as i64);
    //                             pack_b::<T>(
    //                                 b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
    //                                 packed_b,
    //                                 ldb,
    //                                 rhs_col_stride,
    //                                 jb,
    //                                 pb,
    //                                 kc,
    //                                 nr,
    //                                 jj_start,
    //                                 need_full_pack,
    //                             );
    //                             let packed_a = if do_lhs_pack {
    //                                 packed_a
    //                             } else {
    //                                 a + ((i as i64) * lda + (p as i64) * lhs_col_stride)
    //                             };
    //                             for ii in (i_start..ib).step_by(mr) {
    //                                 let mb = min(mr, ib - ii);
    //                                 let micro_kernel = <T>::get_kernel(nr / <T>::Vec::SIZE, mb);
    //                                 for jj in (jj_start..jb).step_by(nr) {
    //                                     let jjb = min(nr, jb - jj);
    //                                     let packed_b = packed_b + (jj as i64) * (kc as i64);
    //                                     if do_lhs_pack {
    //                                         call_microkernel!(
    //                                             need_post_op,
    //                                             packed_a + (kc as i64) * (ii as i64),
    //                                             packed_b,
    //                                             c + (ii as i64) * ldc + (jj as i64),
    //                                             micro_kernel,
    //                                             ldc,
    //                                             pb,
    //                                             jjb,
    //                                             mb as i64,
    //                                             first_kiter,
    //                                             i + ii,
    //                                             j + jj,
    //                                             p + pb == k,
    //                                             _post_op,
    //                                             _post_op_vec
    //                                         );
    //                                     } else {
    //                                         call_non_packed_microkernel!(
    //                                             need_post_op,
    //                                             packed_a + (ii as i64) * lda,
    //                                             packed_b,
    //                                             c + (ii as i64) * ldc + (jj as i64),
    //                                             micro_kernel,
    //                                             ldc,
    //                                             lda,
    //                                             pb,
    //                                             jjb,
    //                                             lhs_col_stride,
    //                                             first_kiter,
    //                                             i + ii,
    //                                             j + jj,
    //                                             p + pb == k,
    //                                             _post_op,
    //                                             _post_op_vec
    //                                         );
    //                                     }
    //                                     job_count += 1;
    //                                     if job_count >= use_end {
    //                                         break 'outer;
    //                                     }
    //                                 }
    //                                 jj_start = 0;
    //                             }
    //                             i_start = 0;
    //                         }
    //                         if p < k {
    //                             barrier.wait();
    //                         }
    //                     }
    //                 }
    //             });
    //         },
    //     );
    // });
}

#[duplicate::duplicate_item(
    func_name           ty      get_kernel                                  args;
    [matmul_mp]         [T]     [get_mixed_precision_kernel]                [vec_cast_back, cast_back];
    [matmul_mp_post]    [T]     [get_mixed_precision_kernel_with_post_op]   [p + pb == k, i + ii, j + jj, vec_cast_back, cast_back, _post_op.clone(), _post_op_vec.clone()];
)]
pub(crate) fn func_name<T, IM, F1, F2>(
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
    with_locked: bool,
    _post_op: F1,
    _post_op_vec: F2,
    pack_vec: fn(*mut IM::Vec, *const T::Vec, usize),
    pack_vec_exceed: fn(*mut IM::Vec, usize),
    pack_zero: fn(&mut IM, &T),
    vec_cast_back: fn(*mut T::Vec, *const IM::Vec),
    cast_back: fn(&mut T, &IM)
)
    where
        T: CommonBounds + MatmulMicroKernel,
        IM: CommonBounds,
        F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
        F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static
{
    assert_eq!(
        nr % T::Vec::SIZE,
        0,
        "nr must be a multiple of {} for type {}",
        T::Vec::SIZE,
        T::STR
    );

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;

    let packed_a = L3_SLAB.with(|mem| {
        let mut mem = mem.borrow_mut();
        let stack = DynStack::new(&mut mem);
        let (packed_a_storage, _) = stack.make_aligned_uninit::<IM>(num_mr_blocks * mr * kc, ALIGN);
        let packed_a = Pointer::new(
            packed_a_storage.as_mut_ptr() as *mut IM,
            (num_mr_blocks * mr * kc) as i64
        );
        packed_a
    });

    let mc_jobs = calculate_jobs(n, nc, mr, nr, mc);
    let mc_rem_jobs = calculate_jobs(n, nc, mr, nr, m % mc);
    num_threads = num_threads.min(mc_jobs);
    let jobs_per_thread =
        mc_jobs.div_ceil(num_threads) * mc + ((mc_rem_jobs.div_ceil(num_threads) * m) % mc);
    if jobs_per_thread < 1024 {
        while num_threads > 1 {
            num_threads -= 1;
            let jobs_per_thread =
                mc_jobs.div_ceil(num_threads) * mc + ((mc_rem_jobs.div_ceil(num_threads) * m) % mc);
            if jobs_per_thread >= 1024 {
                break;
            }
        }
    }
    let mb_per_thread = num_mr_blocks.div_ceil(num_threads);
    let intervals = mt_intervals(mc_jobs, num_threads);
    let mc_rem_intervals = mt_intervals(mc_rem_jobs, num_threads);
    let prgs = calculate_prgs(n, nc, mr, nr, mc, &intervals);
    let rem_prgs = calculate_prgs(n, nc, mr, nr, m % mc, &mc_rem_intervals);

    // if num_threads == 1 {
    //     let mut i = 0;
    //     while i < m {
    //         let ib = min(mc, m - i);
    //         let mut p = 0;
    //         while p < k {
    //             let first_kiter = p == 0;
    //             let pb = min(kc, k - p);
    //             pack_a_mixed_precision::<T, IM>(
    //                 a.clone() + (i as i64) * lda + (p as i64) * lhs_col_stride,
    //                 packed_a.clone(),
    //                 lda,
    //                 lhs_col_stride,
    //                 ib,
    //                 pb,
    //                 kc,
    //                 mr,
    //                 0,
    //                 mb_per_thread,
    //                 num_mr_blocks,
    //                 pack_zero
    //             );
    //             let use_prg = if ib == mc { prgs[0] } else { rem_prgs[0] };
    //             let use_start = if ib == mc { intervals[0].0 } else { mc_rem_intervals[0].0 };
    //             let use_end = if ib == mc { intervals[0].1 } else { mc_rem_intervals[0].1 };
    //             let j_start = use_prg[0] * nc;
    //             let mut job_count = use_start;
    //             let mut i_start = use_prg[1] * mr;
    //             let mut jj_start = use_prg[2] * nr;
    //             L2_SLAB.with(|mem| {
    //                 let mut mem = mem.borrow_mut();
    //                 let stack = DynStack::new(&mut mem);
    //                 let (packed_b_storage, _) = stack.make_aligned_uninit::<IM>(
    //                     num_nr_blocks * nr * kc,
    //                     ALIGN
    //                 );
    //                 let packed_b = Pointer::new(
    //                     packed_b_storage.as_mut_ptr() as *mut IM,
    //                     (num_nr_blocks * nr * kc) as i64
    //                 );
    //                 'outer: for j in (j_start..n).step_by(nc) {
    //                     let jb = min(nc, n - j);
    //                     let c = out.clone() + (i as i64) * ldc + (j as i64);
    //                     pack_b_mixed_precision::<T, IM>(
    //                         b.clone() + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
    //                         packed_b.clone(),
    //                         ldb,
    //                         rhs_col_stride,
    //                         jb,
    //                         pb,
    //                         kc,
    //                         nr,
    //                         pack_vec,
    //                         pack_vec_exceed,
    //                         pack_zero
    //                     );
    //                     let packed_a = packed_a.clone();
    //                     for ii in (i_start..ib).step_by(mr) {
    //                         let mb = min(mr, ib - ii);
    //                         let micro_kernel = <T>::get_kernel(nr / <T>::Vec::SIZE, mb);

    //                         for jj in (jj_start..jb).step_by(nr) {
    //                             let jjb = min(nr, jb - jj);
    //                             micro_kernel(
    //                                 packed_a.clone() + (kc as i64) * (ii as i64),
    //                                 packed_b.clone() + (jj as i64) * (kc as i64),
    //                                 c.clone() + (ii as i64) * ldc + (jj as i64),
    //                                 ldc,
    //                                 1,
    //                                 kc,
    //                                 jjb,
    //                                 mb as i64,
    //                                 first_kiter,
    //                                 args
    //                             );
    //                             job_count += 1;
    //                             if job_count >= use_end {
    //                                 break 'outer;
    //                             }
    //                         }
    //                         jj_start = 0;
    //                     }
    //                     i_start = 0;
    //                 }
    //             });
    //             p += kc;
    //         }
    //         i += mc;
    //     }
    // } else {
        let func = || {
            let mut i = 0;
            while i < m {
                let ib = min(mc, m - i);
                let mut p = 0;
                while p < k {
                    let first_kiter = p == 0;
                    let pb = min(kc, k - p);
                    spindle::for_each_raw(num_threads, |tid| {
                        pack_a_mixed_precision::<T, IM>(
                            a.clone() + (i as i64) * lda + (p as i64) * lhs_col_stride,
                            packed_a.clone(),
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
                        let use_prg = if ib == mc { prgs[tid] } else { rem_prgs[tid] };
                        let use_start = if ib == mc {
                            intervals[tid].0
                        } else {
                            mc_rem_intervals[tid].0
                        };
                        let use_end = if ib == mc {
                            intervals[tid].1
                        } else {
                            mc_rem_intervals[tid].1
                        };
                        let j_start = use_prg[0] * nc;
                        let mut job_count = use_start;
                        let mut i_start = use_prg[1] * mr;
                        let mut jj_start = use_prg[2] * nr;
                        L2_SLAB.with(|mem| {
                            let mut mem = mem.borrow_mut();
                            let stack = DynStack::new(&mut mem);
                            let (packed_b_storage, _) = stack.make_aligned_uninit::<IM>(
                                num_nr_blocks * nr * kc,
                                ALIGN
                            );
                            let packed_b = Pointer::new(
                                packed_b_storage.as_mut_ptr() as *mut IM,
                                (num_nr_blocks * nr * kc) as i64
                            );
                            'outer: for j in (j_start..n).step_by(nc) {
                                let jb = min(nc, n - j);
                                let c = out.clone() + (i as i64) * ldc + (j as i64);
                                pack_b_mixed_precision::<T, IM>(
                                    b.clone() + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                                    packed_b.clone(),
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
                                let packed_a = packed_a.clone();
                                for ii in (i_start..ib).step_by(mr) {
                                    let mb = min(mr, ib - ii);
                                    let micro_kernel = <T>::get_kernel(nr / <T>::Vec::SIZE, mb);

                                    for jj in (jj_start..jb).step_by(nr) {
                                        let jjb = min(nr, jb - jj);
                                        micro_kernel(
                                            packed_a.clone() + (kc as i64) * (ii as i64),
                                            packed_b.clone() + (jj as i64) * (kc as i64),
                                            c.clone() + (ii as i64) * ldc + (jj as i64),
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
                    });
                    p += kc;
                }
                i += mc;
            }
        };
        if with_locked {
            func();
        } else {
            spindle::with_lock(num_threads, func);
        }
    // }

    // THREAD_POOL.with_borrow(|pool| {
    //     pool.parallel_for(
    //         (0..num_threads)
    //             .into_iter()
    //             .zip(prgs.into_iter())
    //             .zip(rem_prgs.into_iter())
    //             .zip(intervals.into_iter())
    //             .zip(mc_rem_intervals.into_iter()),
    //         move |((((tid, prg), rem_prg), (start, end)), (start_rem, end_rem)), _| {
    //             L2_SLAB.with(|mem| {
    //                 let mut mem = mem.borrow_mut();
    //                 let stack = DynStack::new(&mut mem);
    //                 let (packed_b_storage, _) = stack.make_aligned_uninit::<IM>(
    //                     num_nr_blocks * nr * kc,
    //                     ALIGN
    //                 );
    //                 let packed_b = Pointer::new(
    //                     packed_b_storage.as_mut_ptr() as *mut IM,
    //                     (num_nr_blocks * nr * kc) as i64
    //                 );
    //                 let mut i = 0;
    //                 while i < m {
    //                     let ib = min(mc, m - i);
    //                     let use_prg = if ib == mc { prg } else { rem_prg };
    //                     let use_start = if ib == mc { start } else { start_rem };
    //                     let use_end = if ib == mc { end } else { end_rem };
    //                     let j_start = use_prg[0] * nc;
    //                     let mut p = 0;
    //                     while p < k {
    //                         let first_kiter = p == 0;
    //                         let pb = min(kc, k - p);
    //                         pack_a_mixed_precision::<T, IM>(
    //                             a.clone() + (i as i64) * lda + (p as i64) * lhs_col_stride,
    //                             packed_a.clone(),
    //                             lda,
    //                             lhs_col_stride,
    //                             ib,
    //                             pb,
    //                             kc,
    //                             mr,
    //                             tid,
    //                             mb_per_thread,
    //                             num_mr_blocks,
    //                             pack_zero
    //                         );
    //                         barrier.wait();

    //                         let mut job_count = use_start;
    //                         let mut i_start = use_prg[1] * mr;
    //                         let mut jj_start = use_prg[2] * nr;
    //                         'outer: for j in (j_start..n).step_by(nc) {
    //                             let jb = min(nc, n - j);
    //                             let c = out.clone() + (i as i64) * ldc + (j as i64);
    //                             pack_b_mixed_precision::<T, IM>(
    //                                 b.clone() + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
    //                                 packed_b.clone(),
    //                                 ldb,
    //                                 rhs_col_stride,
    //                                 jb,
    //                                 pb,
    //                                 kc,
    //                                 nr,
    //                                 pack_vec,
    //                                 pack_vec_exceed,
    //                                 pack_zero
    //                             );
    //                             let packed_a = packed_a.clone();
    //                             for ii in (i_start..ib).step_by(mr) {
    //                                 let mb = min(mr, ib - ii);
    //                                 let micro_kernel = <T>::get_kernel(nr / <T>::Vec::SIZE, mb);

    //                                 for jj in (jj_start..jb).step_by(nr) {
    //                                     let jjb = min(nr, jb - jj);
    //                                     micro_kernel(
    //                                         packed_a.clone() + (kc as i64) * (ii as i64),
    //                                         packed_b.clone() + (jj as i64) * (kc as i64),
    //                                         c.clone() + (ii as i64) * ldc + (jj as i64),
    //                                         ldc,
    //                                         1,
    //                                         kc,
    //                                         jjb,
    //                                         mb as i64,
    //                                         first_kiter,
    //                                         args
    //                                     );
    //                                     job_count += 1;
    //                                     if job_count >= use_end {
    //                                         break 'outer;
    //                                     }
    //                                 }
    //                                 jj_start = 0;
    //                             }
    //                             i_start = 0;
    //                         }
    //                         p += kc;
    //                         if p < k {
    //                             barrier.wait();
    //                         }
    //                     }
    //                     i += mc;
    //                 }
    //             });
    //         }
    //     );
    // });
}

#[inline]
pub(crate) fn pack_a<T>(
    a: Pointer<T>,
    mut packed_a: Pointer<T>,
    lda: i64,
    stride: i64,
    mc: usize,
    kb: usize,
    kc: usize,
    mr: usize,
    tid: usize,
    mb_per_thread: usize,
    num_mr_blocks: usize
)
    where T: CommonBounds
{
    let start_block = tid * mb_per_thread;
    let end_block = std::cmp::min((tid + 1) * mb_per_thread, num_mr_blocks);
    if start_block >= num_mr_blocks {
        return;
    }
    let start_i = start_block * mr;
    let end_i = std::cmp::min(end_block * mr, mc);
    let offset = start_block * mr * kc;
    packed_a += offset as i64;
    for i in (start_i..end_i).step_by(mr) {
        let mb = mr.min(mc - i);
        for p in 0..kb as i64 {
            for ii in 0..mb as i64 {
                let row = (i as i64) + ii;
                *packed_a = a[row * lda + p * stride];
                packed_a += 1i64;
            }
        }
        for _ in kb..kc {
            for _ in 0..mb as i64 {
                *packed_a = T::ZERO;
                packed_a += 1i64;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn pack_a_mixed_precision<T, I>(
    a: Pointer<T>,
    mut packed_a: Pointer<I>,
    lda: i64,
    stride: i64,
    mc: usize,
    kb: usize,
    kc: usize,
    mr: usize,
    tid: usize,
    mb_per_thread: usize,
    num_mr_blocks: usize,
    cast: fn(&mut I, &T)
)
    where T: CommonBounds, I: CommonBounds
{
    let start_block = tid * mb_per_thread;
    let end_block = std::cmp::min((tid + 1) * mb_per_thread, num_mr_blocks);
    if start_block >= num_mr_blocks {
        return;
    }
    let start_i = start_block * mr;
    let end_i = std::cmp::min(end_block * mr, mc);
    let offset = start_block * mr * kc;
    packed_a += offset as i64;
    for i in (start_i..end_i).step_by(mr) {
        let mb = mr.min(mc - i);
        for p in 0..kb as i64 {
            for ii in 0..mb as i64 {
                let row = (i as i64) + ii;
                cast(&mut *packed_a, &a[row * lda + p * stride]);
                packed_a += 1i64;
            }
        }
        for _ in kb..kc {
            for _ in 0..mb as i64 {
                *packed_a = I::ZERO;
                packed_a += 1i64;
            }
        }
    }
}

#[inline(never)]
pub(crate) fn pack_b<T>(
    b: Pointer<T>,
    mut packed_b: Pointer<T>,
    ldb: i64,
    stride: i64,
    nc: usize,
    kb: usize,
    kc: usize,
    nr: usize,
    jj_start: usize,
    need_full_pack: bool
)
    where T: CommonBounds
{
    let start = if need_full_pack {
        0
    } else {
        for j in (0..jj_start).step_by(nr) {
            let nb = nr.min(nc - j);
            if nb == nr {
                packed_b += (nr as i64) * (kb as i64);
                packed_b += ((kc - kb) as i64) * (nr as i64);
            } else {
                packed_b += ((nr - nb) as i64) * (kb as i64);
                packed_b += ((kc - kb) as i64) * (nr as i64);
            }
        }
        jj_start
    };
    let nr_div_lane = nr / T::Vec::SIZE;
    for j in (start..nc).step_by(nr) {
        let nb = nr.min(nc - j);
        if nb == nr && stride == 1 {
            for p in 0..kb as i64 {
                unsafe {
                    let b_ptr = b.ptr.offset(((p * ldb) as isize) + (j as isize)) as *const T::Vec;
                    for i in 0..nr_div_lane {
                        let packed_b_vec = packed_b.ptr.add(i * T::Vec::SIZE) as *mut T::Vec;
                        packed_b_vec.write(b_ptr.add(i).read_unaligned());
                    }
                }
                packed_b += nr as i64;
            }
            packed_b += ((kc - kb) as i64) * (nr as i64);
        } else {
            for p in 0..kb as i64 {
                for jj in 0..nb as i64 {
                    let j = (j as i64) + jj;
                    *packed_b = b[p * ldb + j * stride];
                    packed_b += 1i64;
                }
                packed_b += (nr - nb) as i64;
            }
            packed_b += ((kc - kb) as i64) * (nr as i64);
        }
    }
}

#[inline(always)]
pub(crate) fn pack_b_mixed_precision<T, I>(
    b: Pointer<T>,
    mut packed_b: Pointer<I>,
    ldb: i64,
    stride: i64,
    nc: usize,
    kb: usize,
    kc: usize,
    nr: usize,
    pack_vec: fn(*mut I::Vec, *const T::Vec, usize),
    pack_vec_exceed: fn(*mut I::Vec, usize),
    pack_zero: fn(&mut I, &T)
)
    where T: CommonBounds, I: CommonBounds
{
    let nr_div_lane = nr / T::Vec::SIZE;

    for j in (0..nc).step_by(nr) {
        let nb = nr.min(nc - j);
        if nb == nr && stride == 1 {
            for p in 0..kb as i64 {
                for i in 0..nr_div_lane {
                    pack_vec(
                        packed_b.ptr as *mut I::Vec,
                        (unsafe {
                            b.ptr.offset(((p * ldb) as isize) + (j as isize))
                        }) as *const T::Vec,
                        i
                    );
                }
                packed_b += nr as i64;
            }
            for _ in kb..kc {
                for i in 0..nr_div_lane {
                    pack_vec_exceed(packed_b.ptr as *mut I::Vec, i);
                }
                packed_b += nr as i64;
            }
        } else {
            for p in 0..kb as i64 {
                for jj in 0..nb as i64 {
                    let j = (j as i64) + jj;
                    let mut res = I::ZERO;
                    pack_zero(&mut res, &b[p * ldb + j * stride]);
                    *packed_b = res;
                    packed_b += 1i64;
                }
                for _ in nb..nr {
                    *packed_b = I::ZERO;
                    packed_b += 1i64;
                }
            }
            for _ in kb..kc {
                for _ in 0..nb as i64 {
                    *packed_b = I::ZERO;
                    packed_b += 1i64;
                }
                for _ in nb..nr {
                    *packed_b = I::ZERO;
                    packed_b += 1i64;
                }
            }
        }
    }
}
