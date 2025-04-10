use crate::backend::Cpu;
use crate::backends::cpu::kernels::matmul::common::{calculate_jobs, calculate_prgs, L2_SLAB};
use crate::tensor_base::_Tensor;
use crate::ALIGN;
use dyn_stack::DynStack;
use gemm_common::cache::{DivCeil, KernelParams, CACHE_INFO};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::shape::shape_utils::mt_intervals;
use hpt_common::{error::base::TensorError, Pointer};
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::traits::VecTrait;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::cmp::min;

use super::common::matmul_prepare;
use super::microkernel_trait::MatmulMicroKernel;

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

    #[cfg(not(target_feature = "neon"))]
    let mut do_lhs_pack = false;
    #[cfg(target_feature = "neon")]
    let mut do_lhs_pack = true;

    if (lhs_col_stride == 1 && n > 128 * nr) || lhs_col_stride != 1 {
        do_lhs_pack = true;
    }

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;
    let packed_a_layout = std::alloc::Layout::from_size_align(
        num_mr_blocks * mr * kc * std::mem::size_of::<T>(),
        ALIGN,
    )
    .expect("layout create failed");

    let packed_a = if do_lhs_pack {
        let a_buffer = unsafe { std::alloc::alloc(packed_a_layout) };
        #[cfg(feature = "bound_check")]
        let ret = Pointer::new(
            a_buffer as *mut T,
            (packed_a_layout.size() / std::mem::size_of::<T>()) as i64,
        );
        #[cfg(not(feature = "bound_check"))]
        let ret = Pointer::new(a_buffer as *mut T);
        ret
    } else {
        a.clone()
    };

    let packed_a_ptr = packed_a.ptr as *mut T;

    let mc_jobs = calculate_jobs(n, nc, mr, nr, mc);
    let mc_rem_jobs = calculate_jobs(n, nc, mr, nr, m % mc);
    num_threads = num_threads.min(mc_jobs);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(num_threads));
    let mb_per_thread = num_mr_blocks.div_ceil(num_threads);
    let intervals = mt_intervals(mc_jobs, num_threads);
    let mc_rem_intervals = mt_intervals(mc_rem_jobs, num_threads);
    let prgs = calculate_prgs(n, nc, mr, nr, mc, &intervals);
    let rem_prgs = calculate_prgs(n, nc, mr, nr, m % mc, &mc_rem_intervals);
    (0..num_threads)
        .into_par_iter()
        .zip(prgs)
        .zip(rem_prgs)
        .zip(intervals)
        .zip(mc_rem_intervals)
        .for_each(
            |((((tid, prg), rem_prg), (start, end)), (start_rem, end_rem))| {
                L2_SLAB.with(|mem| {
                    let mut mem = mem.borrow_mut();
                    let stack = DynStack::new(&mut mem);
                    let (packed_b_storage, _) =
                        stack.make_aligned_uninit::<T>(num_nr_blocks * nr * kc, ALIGN);
                    #[cfg(feature = "bound_check")]
                    let packed_b = Pointer::new(
                        packed_b_storage.as_mut_ptr() as *mut T,
                        (num_nr_blocks * nr * kc) as i64,
                    );
                    #[cfg(not(feature = "bound_check"))]
                    let packed_b = Pointer::new(packed_b_storage.as_mut_ptr() as *mut T);
                    let mut i = 0;
                    while i < m {
                        let ib = min(mc, m - i);
                        let use_prg = if ib == mc { prg } else { rem_prg };
                        let use_start = if ib == mc { start } else { start_rem };
                        let use_end = if ib == mc { end } else { end_rem };
                        let j_start = use_prg[0] * nc;
                        let mut p = 0;
                        while p < k {
                            let first_kiter = p == 0;
                            let pb = min(kc, k - p);
                            if do_lhs_pack {
                                pack_a::<T>(
                                    a.clone() + i as i64 * lda + p as i64 * lhs_col_stride,
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
                                );
                                barrier.wait();
                            }

                            let mut job_count = use_start;
                            let mut i_start = use_prg[1] * mr;
                            let mut jj_start = use_prg[2] * nr;
                            'outer: for j in (j_start..n).step_by(nc) {
                                let jb = min(nc, n - j);
                                let c = out.clone() + i as i64 * ldc + j as i64;
                                pack_b::<T>(
                                    b.clone() + (p as i64 * ldb + j as i64 * rhs_col_stride),
                                    packed_b.clone(),
                                    ldb,
                                    rhs_col_stride,
                                    jb,
                                    pb,
                                    kc,
                                    nr,
                                );
                                let packed_a = if do_lhs_pack {
                                    packed_a.clone()
                                } else {
                                    a.clone() + (i as i64 * lda + p as i64 * lhs_col_stride)
                                };
                                for i in (i_start..ib).step_by(mr) {
                                    let mb = min(mr, ib - i);
                                    let micro_kernel = <T>::get_kernel(nr / <T>::Vec::SIZE, mb);

                                    for jj in (jj_start..jb).step_by(nr) {
                                        let jjb = min(nr, jb - jj);
                                        if do_lhs_pack {
                                            micro_kernel(
                                                packed_a.clone() + kc as i64 * i as i64,
                                                packed_b.clone() + jj as i64 * kc as i64,
                                                c.clone() + i as i64 * ldc + jj as i64,
                                                ldc,
                                                1,
                                                kc,
                                                jjb,
                                                mb as i64,
                                                first_kiter,
                                            );
                                        } else {
                                            micro_kernel(
                                                packed_a.clone() + i as i64 * lda,
                                                packed_b.clone() + jj as i64 * kc as i64,
                                                c.clone() + i as i64 * ldc + jj as i64,
                                                ldc,
                                                lda,
                                                kc,
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
                            p += kc;
                            if p < k {
                                barrier.wait();
                            }
                        }
                        i += mc;
                    }
                });
            },
        );

    if do_lhs_pack {
        unsafe {
            std::alloc::dealloc(packed_a_ptr as *mut u8, packed_a_layout);
        }
    }
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
    let mr = T::get_max_mr().min(m);
    let mut param = if m <= 64 && n <= 64 {
        // skip expensive kernel_params call for small sizes
        let kc = k.min(512);
        let alloc = CACHE_INFO[1].cache_bytes / core::mem::size_of::<T>();
        let nc = (alloc / kc) / nr * nr;
        KernelParams {
            kc,
            mc: m.msrv_next_multiple_of(mr),
            nc,
        }
    } else {
        gemm_common::cache::kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>())
    };
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
        param.nc,
        param.mc,
        nr,
        mr,
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

#[inline]
pub(crate) fn pack_b<T>(
    b: Pointer<T>,
    mut packed_b: Pointer<T>,
    ldb: i64,
    stride: i64,
    nc: usize,
    kb: usize,
    kc: usize,
    nr: usize,
) where
    T: CommonBounds,
{
    let nr_div_lane = nr / T::Vec::SIZE;
    for j in (0..nc).step_by(nr) {
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
            for _ in kb..kc {
                for i in 0..nr_div_lane {
                    let packed_b_vec = unsafe { packed_b.ptr.add(i * T::Vec::SIZE) } as *mut T::Vec;
                    unsafe { packed_b_vec.write(T::Vec::splat(T::ZERO)) };
                }
                packed_b += nr as i64;
            }
        } else {
            for p in 0..kb as i64 {
                for jj in 0..nb as i64 {
                    let j = j as i64 + jj;
                    *packed_b = b[p * ldb + j * stride];
                    packed_b += 1i64;
                }
                for _ in nb..nr {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
            }
            for _ in kb..kc {
                for _ in 0..nr as i64 {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
            }
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
