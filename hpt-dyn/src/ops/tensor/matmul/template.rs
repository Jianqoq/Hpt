use std::cmp::min;

use dyn_stack::DynStack;
use hpt_common::error::base::TensorError;
use hpt_common::{Pointer, shape::shape_utils::mt_intervals};
use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::ToDType;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::microkernel_trait::MatmulMicroKernel;
use super::utils::PrePackedRhs;
use crate::ops::tensor::matmul::common::{L2_SLAB, L3_SLAB, calculate_jobs, calculate_prgs};
use crate::{ALIGN, Device, Tensor};
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
    func_name           ty           get_kernel                 get_horizontal_kernel                   need_post_op;
    [matmul]            [T]          [get_kernel]               [get_horizontal_kernel]                 [false];
    [matmul_post]       [T]          [get_kernel_with_post_op]  [get_horizontal_kernel_with_post_op]    [true];
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
    T: CommonBounds + MatmulMicroKernel,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
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
            let (packed_a_storage, _) =
                stack.make_aligned_uninit::<T>(num_mr_blocks * mr * kc, ALIGN);
            let packed_a = Pointer::new(
                packed_a_storage.as_mut_ptr() as *mut T,
                (num_mr_blocks * mr * kc) as i64,
            );
            packed_a
        } else {
            a.clone()
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
    let get_kernel = if mr == 1 {
        <T>::get_horizontal_kernel
    } else {
        <T>::get_kernel
    };

    let panel_size = num_nr_blocks * nr * kc;

    // credit to sarah-quinones with help on spindle
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
            let packed_a = if do_lhs_pack {
                packed_a
            } else {
                a + ((i as i64) * lda + (p as i64) * lhs_col_stride)
            };

            spindle::for_each_raw(num_threads, |tid| {
                let use_start = intervals[tid].0;
                let use_end = intervals[tid].1;

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
                            stack.make_aligned_with::<T>(num_nr_blocks * nr * kc, ALIGN, |_| {
                                T::ZERO
                            });
                        let packed_b =
                            Pointer::new(packed_b_storage.as_mut_ptr(), panel_size as i64);
                        packed_b
                    })
                };
                let packed_b_cpy = packed_b;
                let use_prg = prgs[tid];
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
                            need_full_pack,
                        );
                    } else {
                        packed_b = packed_b_cpy + panel_idx * panel_size as i64;
                        panel_idx += 1;
                    }
                    for ii in (i_start..ib).step_by(mr) {
                        let mb = min(mr, ib - ii);
                        let micro_kernel = get_kernel(nr / <T>::Vec::SIZE, mb);
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
                                    _post_op.clone(),
                                    _post_op_vec.clone()
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
                                    _post_op.clone(),
                                    _post_op_vec.clone()
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
    prepack_rhs: Option<PrePackedRhs>,
    _post_op: F1,
    _post_op_vec: F2,
    pack_vec: fn(*mut IM::Vec, *const T::Vec, usize),
    pack_vec_exceed: fn(*mut IM::Vec, usize),
    pack_zero: fn(&mut IM, &T),
    vec_cast_back: fn(*mut T::Vec, *const IM::Vec),
    cast_back: fn(&mut T, &IM),
) where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
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
            (num_mr_blocks * mr * kc) as i64,
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
                    buffer.cast::<IM>()
                } else {
                    L2_SLAB.with(|mem| {
                        let mut mem = mem.borrow_mut();
                        let stack = DynStack::new(&mut mem);
                        let (packed_b_storage, _) =
                            stack.make_aligned_uninit::<IM>(num_nr_blocks * nr * kc, ALIGN);
                        Pointer::new(
                            packed_b_storage.as_mut_ptr() as *mut IM,
                            (num_nr_blocks * nr * kc) as i64,
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
                            pack_zero,
                        );
                    } else {
                        packed_b = packed_b_cpy + panel_idx * panel_size as i64;
                        panel_idx += 1;
                    }
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

pub(crate) fn prepack_b<T>(
    b: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    ldb: i64,
    rhs_col_stride: i64,
    kc: usize,
    mc: usize,
    nc: usize,
    nr: usize,
    mr: usize,
    mut num_threads: usize,
    device: Device,
) -> Result<PrePackedRhs, TensorError>
where
    T: CommonBounds + ToDType,
{
    assert_eq!(
        nr % T::Vec::SIZE,
        0,
        "nr must be a multiple of {} for type {}",
        T::Vec::SIZE,
        T::STR
    );

    let num_nr_blocks = (nc + nr - 1) / nr;

    let mc_jobs = calculate_jobs(n, nc, mr, nr, mc);
    let mc_rem_jobs = calculate_jobs(n, nc, mr, nr, m % mc);
    num_threads = num_threads.min(mc_jobs);
    let intervals = mt_intervals(mc_jobs, num_threads);
    let mc_rem_intervals = mt_intervals(mc_rem_jobs, num_threads);
    let prgs = calculate_prgs(n, nc, mr, nr, mc, &intervals);
    let rem_prgs = calculate_prgs(n, nc, mr, nr, m % mc, &mc_rem_intervals);

    let panel_size = num_nr_blocks * nr * kc;

    let func = |ib: usize| {
        let prgs = if ib == mc { &prgs } else { &rem_prgs };
        let intervals = if ib == mc {
            &intervals
        } else {
            &mc_rem_intervals
        };
        let mut buffer_size = vec![0; num_threads];
        (0..num_threads).for_each(|tid| {
            let use_start = intervals[tid].0;
            let use_end = intervals[tid].1;
            let use_prg = prgs[tid];
            let j_start = use_prg[0] * nc;
            let mut job_count = use_start;
            let mut i_start = use_prg[1] * mr;
            let mut jj_start = use_prg[2] * nr;
            'outer: for j in (j_start..n).step_by(nc) {
                let jb = min(nc, n - j);
                buffer_size[tid] += panel_size;
                for _ in (i_start..ib).step_by(mr) {
                    for _ in (jj_start..jb).step_by(nr) {
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

        let mut cum_buffer_size = vec![0usize; num_threads];
        let mut sum = 0;
        for i in 0..num_threads {
            sum += buffer_size[i];
            cum_buffer_size[i] = sum - buffer_size[i];
        }

        let total_panels_size = buffer_size.iter().sum::<usize>();

        let num_kc = k.div_ceil(kc);
        let buffer = Tensor::zeros(
            &[(total_panels_size * num_kc) as i64],
            T::to_dtype(),
            device.clone(),
        )
        .expect("failed to create buffer");

        let buffer_ptr = buffer.data.cast::<T>();

        let mut buffers = vec![];
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let pb = min(kc, k - p);

            let local_buffers = (0..num_threads)
                .into_par_iter()
                .map(|tid| {
                    let use_start = intervals[tid].0;
                    let use_end = intervals[tid].1;
                    let use_prg = prgs[tid];
                    let j_start = use_prg[0] * nc;
                    let mut job_count = use_start;
                    let mut i_start = use_prg[1] * mr;
                    let mut jj_start = use_prg[2] * nr;
                    let need_full_pack = ib - i_start > mr;

                    let mut buffer_ptr =
                        buffer_ptr + cum_buffer_size[tid] + (p_idx * total_panels_size) as i64;
                    let ptr_cpy = buffer_ptr;
                    'outer: for j in (j_start..n).step_by(nc) {
                        let jb = min(nc, n - j);
                        pack_b::<T>(
                            b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                            buffer_ptr,
                            ldb,
                            rhs_col_stride,
                            jb,
                            pb,
                            kc,
                            nr,
                            jj_start,
                            need_full_pack,
                        );
                        buffer_ptr += panel_size as i64;
                        for _ in (i_start..ib).step_by(mr) {
                            for _ in (jj_start..jb).step_by(nr) {
                                job_count += 1;
                                if job_count >= use_end {
                                    break 'outer;
                                }
                            }
                            jj_start = 0;
                        }
                        i_start = 0;
                    }
                    ptr_cpy.cast::<u8>()
                    // buffer
                })
                .collect::<Vec<_>>();
            buffers.push(local_buffers);
        }
        (buffers, buffer)
    };
    let rem = m % mc;
    if rem > 0 {
        let (buffers_rem, buff_rem_tensor) = func(rem);
        let (buffers, buff_tensor) = func(mc);
        Ok(PrePackedRhs {
            buffers,
            buffer_rems: buffers_rem,
            buffer: buff_tensor,
            buffer_rem: buff_rem_tensor,
            mr,
            mc,
            kc,
            nr,
            nc,
            num_threads,
            prgs,
            rem_prgs,
            intervals,
            rem_intervals: mc_rem_intervals,
        })
    } else {
        let (buffers, buff_tensor) = func(mc);
        Ok(PrePackedRhs {
            buffers: buffers.clone(),
            buffer_rems: buffers,
            buffer: buff_tensor.clone(),
            buffer_rem: buff_tensor,
            mr,
            mc,
            kc,
            nr,
            nc,
            num_threads,
            prgs,
            rem_prgs,
            intervals,
            rem_intervals: mc_rem_intervals,
        })
    }
}

pub(crate) fn prepack_mp_b<T, IM>(
    b: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    ldb: i64,
    rhs_col_stride: i64,
    kc: usize,
    mc: usize,
    nc: usize,
    nr: usize,
    mr: usize,
    mut num_threads: usize,
    device: Device,
    pack_vec: fn(*mut IM::Vec, *const T::Vec, usize),
    pack_vec_exceed: fn(*mut IM::Vec, usize),
    pack_zero: fn(&mut IM, &T),
) -> Result<PrePackedRhs, TensorError>
where
    T: CommonBounds,
    IM: CommonBounds + ToDType,
{
    assert_eq!(
        nr % T::Vec::SIZE,
        0,
        "nr must be a multiple of {} for type {}",
        T::Vec::SIZE,
        T::STR
    );

    let num_nr_blocks = (nc + nr - 1) / nr;

    let mc_jobs = calculate_jobs(n, nc, mr, nr, mc);
    let mc_rem_jobs = calculate_jobs(n, nc, mr, nr, m % mc);
    num_threads = num_threads.min(mc_jobs);
    let intervals = mt_intervals(mc_jobs, num_threads);
    let mc_rem_intervals = mt_intervals(mc_rem_jobs, num_threads);
    let prgs = calculate_prgs(n, nc, mr, nr, mc, &intervals);
    let rem_prgs = calculate_prgs(n, nc, mr, nr, m % mc, &mc_rem_intervals);

    let panel_size = num_nr_blocks * nr * kc;

    let func = |ib: usize| {
        let prgs = if ib == mc { &prgs } else { &rem_prgs };
        let intervals = if ib == mc {
            &intervals
        } else {
            &mc_rem_intervals
        };
        let mut buffer_size = vec![0; num_threads];
        (0..num_threads).for_each(|tid| {
            let use_start = intervals[tid].0;
            let use_end = intervals[tid].1;
            let use_prg = prgs[tid];
            let j_start = use_prg[0] * nc;
            let mut job_count = use_start;
            let mut i_start = use_prg[1] * mr;
            let mut jj_start = use_prg[2] * nr;
            'outer: for j in (j_start..n).step_by(nc) {
                let jb = min(nc, n - j);
                buffer_size[tid] += panel_size;
                for _ in (i_start..ib).step_by(mr) {
                    for _ in (jj_start..jb).step_by(nr) {
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

        let mut cum_buffer_size = vec![0usize; num_threads];
        let mut sum = 0;
        for i in 0..num_threads {
            sum += buffer_size[i];
            cum_buffer_size[i] = sum - buffer_size[i];
        }

        let total_panels_size = buffer_size.iter().sum::<usize>();

        let num_kc = k.div_ceil(kc);
        let buffer = Tensor::zeros(
            &[(total_panels_size * num_kc) as i64],
            IM::to_dtype(),
            device.clone(),
        )
        .expect("failed to create buffer");

        let buffer_ptr = buffer.data.cast::<IM>();

        let mut buffers = vec![];
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let pb = min(kc, k - p);

            let local_buffers = (0..num_threads)
                .into_par_iter()
                .map(|tid| {
                    let use_start = intervals[tid].0;
                    let use_end = intervals[tid].1;
                    let use_prg = prgs[tid];
                    let j_start = use_prg[0] * nc;
                    let mut job_count = use_start;
                    let mut i_start = use_prg[1] * mr;
                    let mut jj_start = use_prg[2] * nr;
                    let mut buffer_ptr =
                        buffer_ptr + cum_buffer_size[tid] + (p_idx * total_panels_size) as i64;
                    let ptr_cpy = buffer_ptr;
                    'outer: for j in (j_start..n).step_by(nc) {
                        let jb = min(nc, n - j);
                        pack_b_mixed_precision::<T, IM>(
                            b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                            buffer_ptr,
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
                        buffer_ptr += panel_size as i64;
                        for _ in (i_start..ib).step_by(mr) {
                            for _ in (jj_start..jb).step_by(nr) {
                                job_count += 1;
                                if job_count >= use_end {
                                    break 'outer;
                                }
                            }
                            jj_start = 0;
                        }
                        i_start = 0;
                    }
                    ptr_cpy.cast::<u8>()
                })
                .collect::<Vec<_>>();
            buffers.push(local_buffers);
        }
        (buffers, buffer)
    };
    let rem = m % mc;
    if rem > 0 {
        let (buffers_rem, buff_rem_tensor) = func(rem);
        let (buffers, buff_tensor) = func(mc);
        Ok(PrePackedRhs {
            buffers,
            buffer_rems: buffers_rem,
            buffer: buff_tensor,
            buffer_rem: buff_rem_tensor,
            mr,
            mc,
            kc,
            nr,
            nc,
            num_threads,
            prgs,
            rem_prgs,
            intervals,
            rem_intervals: mc_rem_intervals,
        })
    } else {
        let (buffers, buff_tensor) = func(mc);
        Ok(PrePackedRhs {
            buffers: buffers.clone(),
            buffer_rems: buffers,
            buffer: buff_tensor.clone(),
            buffer_rem: buff_tensor,
            mr,
            mc,
            kc,
            nr,
            nc,
            num_threads,
            prgs,
            rem_prgs,
            intervals,
            rem_intervals: mc_rem_intervals,
        })
    }
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
    num_mr_blocks: usize,
) where
    T: CommonBounds,
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
    cast: fn(&mut I, &T),
) where
    T: CommonBounds,
    I: CommonBounds,
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
    need_full_pack: bool,
) -> usize
where
    T: CommonBounds,
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
    let origin = packed_b.addr();
    for j in (start..nc).step_by(nr) {
        let nb = nr.min(nc - j);
        if nb == nr && stride == 1 {
            for p in 0..kb as i64 {
                unsafe {
                    let b_ptr = b
                        .offset(((p * ldb) as isize) + (j as isize))
                        .cast::<T::Vec>();
                    for i in 0..nr_div_lane {
                        let packed_b_vec = packed_b.ptr.add(i * T::Vec::SIZE) as *mut T::Vec;
                        packed_b_vec.write((b_ptr + i).read_unaligned());
                    }
                }
                packed_b += nr as i64;
            }
            packed_b += ((kc - kb) as i64) * (nr as i64);
        } else {
            for p in 0..kb as i64 {
                for jj in 0..nb as i64 {
                    *packed_b = b[p * ldb + (j as i64 + jj) * stride];
                    packed_b += 1i64;
                }
                packed_b += (nr - nb) as i64;
            }
            packed_b += ((kc - kb) as i64) * (nr as i64);
        }
    }
    (packed_b.addr() - origin) / std::mem::size_of::<T>()
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
    pack_zero: fn(&mut I, &T),
) where
    T: CommonBounds,
    I: CommonBounds,
{
    let nr_div_lane = nr / T::Vec::SIZE;

    for j in (0..nc).step_by(nr) {
        let nb = nr.min(nc - j);
        if nb == nr && stride == 1 {
            for p in 0..kb as i64 {
                for i in 0..nr_div_lane {
                    pack_vec(
                        packed_b.ptr as *mut I::Vec,
                        (unsafe { b.ptr.offset(((p * ldb) as isize) + (j as isize)) })
                            as *const T::Vec,
                        i,
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
