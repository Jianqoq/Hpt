use std::cmp::min;
use std::sync::atomic::AtomicUsize;

use crate::backend::Cpu;
use crate::backends::cpu::cache_utils::cache::Cache;
use crate::tensor_base::_Tensor;
use crate::{Tensor, ALIGN};
use dyn_stack::DynStack;
use gemm_common::cache::CACHE_INFO;
use gemm_common::gemm::CACHELINE_ALIGN;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape::Shape;
use hpt_common::shape::shape_utils::predict_broadcast_shape;
use hpt_common::shape::shape_utils::{compare_and_pad_shapes, mt_intervals};
use hpt_common::{error::base::TensorError, Pointer};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::traits::VecTrait;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use super::microkernel_trait::MicroKernel;

pub static TLB_L1_SIZE: AtomicUsize = AtomicUsize::new(64);

fn estimate_tlb_miss(
    mr: usize,
    nr: usize,
    kc: usize,
    element_size: usize,
    tlb_entries: usize,
    page_size: usize,
) -> bool {
    let lhs_memory = mr * kc * element_size;
    let rhs_memory = nr * kc * element_size;
    let dst_memory = mr * nr * element_size;
    let total_memory = lhs_memory + rhs_memory + dst_memory;

    let pages_needed = total_memory.div_ceil(page_size);

    pages_needed > tlb_entries
}

pub(crate) fn gemm_prepare<T, const DEVICE: usize, A>(
    lhs: &_Tensor<T, Cpu, DEVICE, A>,
    rhs: &_Tensor<T, Cpu, DEVICE, A>,
    out: Option<_Tensor<T, Cpu, DEVICE, A>>,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: CommonBounds,
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(
                &Shape::from([lhs.shape()[0], rhs.shape()[1]]),
                &out.layout(),
            )?;
            Ok(out)
        } else {
            _Tensor::<T, Cpu, DEVICE, A>::empty(vec![lhs.shape()[0], rhs.shape()[1]])
        };
        res
    } else {
        let (longer_shape, padded_short_shape) = compare_and_pad_shapes(&lhs.shape(), &rhs.shape());
        let a_shape;
        let b_shape;
        if lhs.shape().len() > rhs.shape().len() {
            a_shape = longer_shape;
            b_shape = padded_short_shape;
        } else {
            a_shape = padded_short_shape;
            b_shape = longer_shape;
        }
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let mut res_shape =
            predict_broadcast_shape(&a_shape[..a_shape.len() - 2], &b_shape[..b_shape.len() - 2])?
                .to_vec();
        res_shape.push(a_shape[a_shape.len() - 2]);
        res_shape.push(b_shape[b_shape.len() - 1]);
        let res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out.layout())?;
            Ok(out)
        } else {
            _Tensor::<T, Cpu, DEVICE, A>::empty(res_shape)
        };
        res
    }
}

thread_local! {
    pub static L2_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(dyn_stack::MemBuffer::new(
        dyn_stack::StackReq::new_aligned::<u8>(CACHE_INFO[1].cache_bytes, CACHELINE_ALIGN)
    ));
}

fn calculate_jobs(n: usize, nc: usize, mr: usize, nr: usize, ib: usize) -> usize {
    let mut jobs = 0;
    for j in (0..n).step_by(nc) {
        let jb = min(nc, n - j);
        for _ in (0..ib).step_by(mr) {
            for _ in (0..jb).step_by(nr) {
                jobs += 1;
            }
        }
    }
    jobs
}

fn calculate_prg(
    n: usize,
    nc: usize,
    mr: usize,
    nr: usize,
    ib: usize,
    prg: [usize; 3],
    mut start: usize,
    end: usize,
) -> [usize; 3] {
    let mut ret = prg;
    let j_start = prg[0] * nc;
    let mut i_start = prg[1] * mr;
    let mut jj_start = prg[2] * nr;
    for j in (j_start..n).step_by(nc) {
        let jb = min(nc, n - j);
        for _ in (i_start..ib).step_by(mr) {
            for _ in (jj_start..jb).step_by(nr) {
                ret[2] += 1;
                start += 1;
                if start >= end {
                    return ret;
                }
            }
            ret[1] += 1;
            ret[2] = 0;
            jj_start = 0;
        }
        ret[0] += 1;
        ret[1] = 0;
        ret[2] = 0;
        i_start = 0;
    }
    ret
}

#[allow(unused_labels)]
pub(crate) fn gemm_template<T>(
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
    num_threads: usize,
) where
    T: CommonBounds + MicroKernel + PartialEq,
{
    assert_eq!(
        nr % T::Vec::SIZE,
        0,
        "nr must be a multiple of {} for type {}",
        T::Vec::SIZE,
        T::STR
    );
    assert_eq!(nc % nr, 0, "nc must be a multiple of nr");
    assert_eq!(mc % mr, 0, "mc must be a multiple of mr");

    let mut do_lhs_pack = false;

    if estimate_tlb_miss(
        mr,
        nr,
        kc,
        std::mem::size_of::<T>(),
        TLB_L1_SIZE.load(std::sync::atomic::Ordering::Relaxed),
        4096,
    ) || (lhs_col_stride == 1 && n > 128 * nr)
        || lhs_col_stride != 1
    {
        do_lhs_pack = true;
    }

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;
    let packed_a_layout = std::alloc::Layout::from_size_align(
        num_mr_blocks * mr * kc * std::mem::size_of::<T>(),
        ALIGN,
    )
    .expect("layout create failed");
    let mb_per_thread = num_mr_blocks.div_ceil(num_threads);

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

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(num_threads));

    let mut prgs = vec![[0, 0, 0]; num_threads];
    let mc_jobs = calculate_jobs(n, nc, mr, nr, mc);
    let mc_rem_jobs = calculate_jobs(n, nc, mr, nr, m % mc);
    let intervals = mt_intervals(mc_jobs, num_threads);
    let mc_rem_intervals = mt_intervals(mc_rem_jobs, num_threads);

    let mut prg = [0, 0, 0];
    for tid in 0..num_threads {
        let (start, end) = intervals[tid];
        prgs[tid] = prg;
        prg = calculate_prg(n, nc, mr, nr, mc, prg, start, end);
    }
    let mut rem_prgs = vec![[0, 0, 0]; num_threads];
    let mut rem_prg = [0, 0, 0];
    for tid in 0..num_threads {
        let (start, end) = mc_rem_intervals[tid];
        rem_prgs[tid] = rem_prg;
        rem_prg = calculate_prg(n, nc, mr, nr, m % mc, rem_prg, start, end);
    }
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
                                    b.clone() + (p as i64 * ldb + j as i64),
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
                                                1,
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
    // for i in (0..mc).step_by(mr) {
    //     let mb = mr.min(mc - i);
    //     for p in 0..kb as i64 {
    //         for ii in 0..mb as i64 {
    //             let i = i as i64 + ii;
    //             *packed_a = a[i * lda + p * stride];
    //             packed_a += 1i64;
    //         }
    //     }
    //     for _ in kb..kc {
    //         for _ in 0..mb as i64 {
    //             *packed_a = T::ZERO;
    //             packed_a += 1i64;
    //         }
    //     }
    // }
}

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
                for _ in 0..nb as i64 {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
                for _ in nb..nr {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
            }
        }
    }
}

/// gemm
pub fn gemm<T, const DEVICE: usize, A>(
    a: &Tensor<T, Cpu, DEVICE, A>,
    b: &Tensor<T, Cpu, DEVICE, A>,
    out: Option<Tensor<T, Cpu, DEVICE, A>>,
    num_threads: usize,
) -> Result<Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: CommonBounds + MicroKernel + PartialEq,
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    let c = gemm_prepare(&a.inner, &b.inner, out.map(|t| t.inner.as_ref().clone()))?;
    let m = a.shape()[0] as usize;
    let n = b.shape()[1] as usize;
    let k = a.shape()[1] as usize;

    let cache = Cache::<T>::new();
    let nr = cache.l1_line_size;
    let mr = T::get_max_mr();
    let param = gemm_common::cache::kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>());
    gemm_template::<T>(
        a.ptr(),
        b.ptr(),
        c.ptr(),
        m,
        n,
        k,
        a.strides()[a.ndim() - 2],
        b.strides()[b.ndim() - 2],
        c.shape()[1] as i64,
        a.strides()[a.ndim() - 1],
        b.strides()[b.ndim() - 1],
        param.kc,
        param.nc,
        param.mc,
        nr,
        T::get_max_mr(),
        num_threads,
    );
    Ok(c.into())
}
