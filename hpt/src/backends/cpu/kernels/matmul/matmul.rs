use crate::backend::Cpu;
use crate::backends::cpu::kernels::matmul::common::{
    calculate_jobs, calculate_prgs, L2_SLAB, L3_SLAB,
};
use crate::tensor_base::_Tensor;
use crate::{ALIGN, CUSTOM_THREAD_POOL};
use dyn_stack::DynStack;
use gemm_common::cache::DivCeil;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::shape::shape_utils::mt_intervals;
use hpt_common::{error::base::TensorError, Pointer};
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::traits::VecTrait;
use std::cmp::min;
use std::sync::Arc;

use super::common::matmul_prepare;
use super::microkernel_trait::MatmulMicroKernel;
use super::utils::kernel_params;

/// single batch matmul template
///
/// # Arguments
///
/// * `a`: lhs shape `(m, k)`
/// * `b`: rhs shape `(k, n)`
/// * `out`: output shape `(m, n)`
/// * `m`: rows of lhs
/// * `n`: cols of rhs
/// * `k`: cols of lhs
/// * `lda`: `lhs.strides[a.ndim() - 2]`
/// * `ldb`: `rhs.strides[r.ndim() - 2]`
/// * `ldc`: `out.shape[out.ndim() - 1]`
/// * `lhs_col_stride`: `lhs.strides[a.ndim() - 1]`
/// * `rhs_col_stride`: `rhs.strides[b.ndim() - 1]`
/// * `kc`: k block size
/// * `mc`: m block size
/// * `nc`: n block size
/// * `nr`: n register block size
/// * `mr`: m register block size
/// * `num_threads`: number of threads
#[inline]
pub fn matmul_template<T>(
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
) where
    T: CommonBounds + MatmulMicroKernel,
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
            #[cfg(feature = "bound_check")]
            let packed_a = Pointer::new(
                packed_a_storage.as_mut_ptr() as *mut T,
                (num_mr_blocks * mr * kc) as i64,
            );
            #[cfg(not(feature = "bound_check"))]
            let packed_a = Pointer::new(packed_a_storage.as_mut_ptr() as *mut T);
            packed_a
        } else {
            a.clone()
        }
    });

    let mc_jobs = calculate_jobs(n, nc, mr, nr, mc);
    let mc_rem_jobs = calculate_jobs(n, nc, mr, nr, m % mc);
    num_threads = num_threads.min(mc_jobs);
    let jobs_per_thread =
        mc_jobs.div_ceil(num_threads) * mc + mc_rem_jobs.div_ceil(num_threads) * m % mc;
    if jobs_per_thread < 1024 {
        while num_threads > 1 {
            num_threads -= 1;
            let jobs_per_thread =
                mc_jobs.div_ceil(num_threads) * mc + mc_rem_jobs.div_ceil(num_threads) * m % mc;
            if jobs_per_thread >= 1024 {
                break;
            }
        }
    }
    let barrier = Arc::new(std::sync::Barrier::new(num_threads));
    let mb_per_thread = num_mr_blocks.div_ceil(num_threads);
    let intervals = mt_intervals(mc_jobs, num_threads);
    let mc_rem_intervals = mt_intervals(mc_rem_jobs, num_threads);
    let prgs = calculate_prgs(n, nc, mr, nr, mc, &intervals);
    let rem_prgs = calculate_prgs(n, nc, mr, nr, m % mc, &mc_rem_intervals);
    CUSTOM_THREAD_POOL.with_borrow(|pool| {
        pool.parallel_for(
            (0..num_threads)
                .into_iter()
                .zip(prgs.into_iter())
                .zip(rem_prgs.into_iter())
                .zip(intervals.into_iter())
                .zip(mc_rem_intervals.into_iter()),
            move |((((tid, prg), rem_prg), (start, end)), (start_rem, end_rem)), _| {
                L2_SLAB.with_borrow_mut(|mem| {
                    let stack = DynStack::new(mem);
                    let (mut packed_b_storage, _) =
                        stack.make_aligned_with::<T>(num_nr_blocks * nr * kc, ALIGN, |_| T::ZERO);
                    #[cfg(feature = "bound_check")]
                    let packed_b = Pointer::new(
                        packed_b_storage.as_mut_ptr() as *mut T,
                        (num_nr_blocks * nr * kc) as i64,
                    );
                    #[cfg(not(feature = "bound_check"))]
                    let packed_b = Pointer::new(packed_b_storage.as_mut_ptr() as *mut T);
                    for i in (0..m).step_by(mc) {
                        let ib = min(mc, m - i);
                        let use_prg = if ib == mc { prg } else { rem_prg };
                        let use_start = if ib == mc { start } else { start_rem };
                        let use_end = if ib == mc { end } else { end_rem };
                        let j_start = use_prg[0] * nc;
                        for p in (0..k).step_by(kc) {
                            let first_kiter = p == 0;
                            let pb = min(kc, k - p);
                            if do_lhs_pack {
                                pack_a::<T>(
                                    a + i as i64 * lda + p as i64 * lhs_col_stride,
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
                                barrier.wait();
                            }

                            let mut job_count = use_start;
                            let mut i_start = use_prg[1] * mr;
                            let mut jj_start = use_prg[2] * nr;
                            let need_full_pack = (ib - i_start) > mr;
                            'outer: for j in (j_start..n).step_by(nc) {
                                let jb = min(nc, n - j);
                                let c = out + i as i64 * ldc + j as i64;
                                pack_b::<T>(
                                    b + (p as i64 * ldb + j as i64 * rhs_col_stride),
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
                                let packed_a = if do_lhs_pack {
                                    packed_a
                                } else {
                                    a + (i as i64 * lda + p as i64 * lhs_col_stride)
                                };
                                for i in (i_start..ib).step_by(mr) {
                                    let mb = min(mr, ib - i);
                                    let micro_kernel = <T>::get_kernel(nr / <T>::Vec::SIZE, mb);
                                    for jj in (jj_start..jb).step_by(nr) {
                                        let jjb = min(nr, jb - jj);
                                        let packed_b = packed_b + jj as i64 * kc as i64;
                                        if do_lhs_pack {
                                            micro_kernel(
                                                packed_a + kc as i64 * i as i64,
                                                packed_b,
                                                c + i as i64 * ldc + jj as i64,
                                                ldc,
                                                1,
                                                pb,
                                                jjb,
                                                mb as i64,
                                                first_kiter,
                                            );
                                        } else {
                                            micro_kernel(
                                                packed_a + i as i64 * lda,
                                                packed_b,
                                                c + i as i64 * ldc + jj as i64,
                                                ldc,
                                                lda,
                                                pb,
                                                jjb,
                                                lhs_col_stride,
                                                first_kiter,
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
                            if p < k {
                                barrier.wait();
                            }
                        }
                    }
                });
            },
        );
    });
}
/// single batch matmul template no block info
///
/// # Arguments
///
/// * `a`: lhs shape `(m, k)`
/// * `b`: rhs shape `(k, n)`
/// * `out`: output shape `(m, n)`
/// * `m`: rows of lhs
/// * `n`: cols of rhs
/// * `k`: cols of lhs
/// * `lda`: `lhs.strides[a.ndim() - 2]`
/// * `ldb`: `rhs.strides[r.ndim() - 2]`
/// * `ldc`: `out.shape[out.ndim() - 1]`
/// * `lhs_col_stride`: `lhs.strides[a.ndim() - 1]`
/// * `rhs_col_stride`: `rhs.strides[b.ndim() - 1]`
/// * `num_threads`: number of threads
#[inline]
pub fn matmul_template_no_block_info<T>(
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
    num_threads: usize,
) where
    T: CommonBounds + MatmulMicroKernel,
{
    let nr = T::get_max_nr() * T::Vec::SIZE;
    let mr = T::get_max_mr();
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
    matmul_template::<T>(
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
        param.kc,
        param.mc,
        param.nc,
        nr,
        mr,
        do_lhs_pack,
        num_threads,
    );
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
                let row = i as i64 + ii;
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
) where
    T: CommonBounds,
{
    let start = if need_full_pack {
        0
    } else {
        for j in (0..jj_start).step_by(nr) {
            let nb = nr.min(nc - j);
            if nb == nr {
                packed_b += nr as i64 * kb as i64;
                packed_b += (kc - kb) as i64 * nr as i64;
            } else {
                packed_b += (nr - nb) as i64 * kb as i64;
                packed_b += (kc - kb) as i64 * nr as i64;
            }
        }
        jj_start
    };
    let nr_div_lane = nr / T::Vec::SIZE;
    for j in (start..nc).step_by(nr) {
        let nb = nr.min(nc - j);
        if nb == nr && stride == 1 {
            for p in 0..kb as i64 {
                for i in 0..nr_div_lane {
                    let packed_b_vec = unsafe { packed_b.ptr.add(i * T::Vec::SIZE) } as *mut T::Vec;
                    unsafe {
                        packed_b_vec.write(
                            (b.ptr.offset(
                                (p * ldb) as isize + (i * T::Vec::SIZE) as isize + j as isize,
                            ) as *const T::Vec)
                                .read_unaligned(),
                        )
                    };
                }
                packed_b += nr as i64;
            }
            packed_b += (kc - kb) as i64 * nr as i64;
        } else {
            for p in 0..kb as i64 {
                for jj in 0..nb as i64 {
                    let j = j as i64 + jj;
                    *packed_b = b[p * ldb + j * stride];
                    packed_b += 1i64;
                }
                packed_b += (nr - nb) as i64;
            }
            packed_b += (kc - kb) as i64 * nr as i64;
        }
    }
}

/// matmul
pub(crate) fn matmul<T, const DEVICE: usize, A>(
    a: &_Tensor<T, Cpu, DEVICE, A>,
    b: &_Tensor<T, Cpu, DEVICE, A>,
    out: Option<_Tensor<T, Cpu, DEVICE, A>>,
    num_threads: usize,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: CommonBounds + MatmulMicroKernel,
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    let c = matmul_prepare(&a, &b, out)?;
    let m = a.shape()[0] as usize;
    let n = b.shape()[1] as usize;
    let k = a.shape()[1] as usize;
    matmul_template_no_block_info::<T>(
        a.ptr(),
        b.ptr(),
        c.ptr(),
        m,
        n,
        k,
        a.strides()[a.ndim() - 2],
        b.strides()[b.ndim() - 2],
        c.strides()[c.ndim() - 2] as i64,
        a.strides()[a.ndim() - 1],
        b.strides()[b.ndim() - 1],
        num_threads,
    );
    Ok(c.into())
}
