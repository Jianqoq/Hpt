use std::cmp::min;

use crate::{
    backends::cpu::kernels::matmul::common::{
        calculate_jobs, calculate_prgs, matmul_prepare, pack_a_mixed_precision,
        pack_b_mixed_precision, L2_SLAB,
    },
    tensor_base::_Tensor,
    ALIGN,
};
use dyn_stack::DynStack;
use gemm_common::cache::KernelParams;
use gemm_common::cache::{DivCeil, CACHE_INFO};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_common::{error::base::TensorError, shape::shape_utils::mt_intervals, Pointer};
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::{dtype::TypeCommon, into_scalar::Cast, traits::VecTrait};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

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
#[inline(always)]
pub fn matmul_mixed_precision_template<T, IM>(
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
    pack_vec: fn(*mut IM::Vec, *const T::Vec, usize),
    pack_vec_exceed: fn(*mut IM::Vec, usize),
    pack_zero: fn(T) -> IM,
    vec_cast_back: fn(*const IM::Vec) -> T::Vec,
    cast_back: fn(IM) -> T,
) where
    T: CommonBounds + MatmulMicroKernel + Cast<IM>,
    IM: CommonBounds,
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
    let packed_a_layout = std::alloc::Layout::from_size_align(
        num_mr_blocks * mr * kc * std::mem::size_of::<IM>(),
        ALIGN,
    )
    .expect("layout create failed");

    let packed_a = {
        let a_buffer = unsafe { std::alloc::alloc(packed_a_layout) };
        #[cfg(feature = "bound_check")]
        let ret = Pointer::new(
            a_buffer as *mut IM,
            (packed_a_layout.size() / std::mem::size_of::<IM>()) as i64,
        );
        #[cfg(not(feature = "bound_check"))]
        let ret = Pointer::new(a_buffer as *mut IM);
        ret
    };

    let packed_a_ptr = packed_a.ptr as *mut IM;

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
                        stack.make_aligned_uninit::<IM>(num_nr_blocks * nr * kc, ALIGN);
                    #[cfg(feature = "bound_check")]
                    let packed_b = Pointer::new(
                        packed_b_storage.as_mut_ptr() as *mut IM,
                        (num_nr_blocks * nr * kc) as i64,
                    );
                    #[cfg(not(feature = "bound_check"))]
                    let packed_b = Pointer::new(packed_b_storage.as_mut_ptr() as *mut IM);
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
                            pack_a_mixed_precision::<T, IM>(
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

                            let mut job_count = use_start;
                            let mut i_start = use_prg[1] * mr;
                            let mut jj_start = use_prg[2] * nr;
                            'outer: for j in (j_start..n).step_by(nc) {
                                let jb = min(nc, n - j);
                                let c = out.clone() + i as i64 * ldc + j as i64;
                                pack_b_mixed_precision::<T, IM>(
                                    b.clone() + (p as i64 * ldb + j as i64),
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
                                let packed_a = packed_a.clone();
                                for i in (i_start..ib).step_by(mr) {
                                    let mb = min(mr, ib - i);
                                    let micro_kernel =
                                        <T>::get_mixed_precision_kernel(nr / <T>::Vec::SIZE, mb);

                                    for jj in (jj_start..jb).step_by(nr) {
                                        let jjb = min(nr, jb - jj);
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
                                            vec_cast_back,
                                            cast_back,
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

    unsafe {
        std::alloc::dealloc(packed_a_ptr as *mut u8, packed_a_layout);
    }
}

/// single batch matmul mixed precision template no block info
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
#[inline(always)]
pub fn matmul_mixed_precision_template_no_block_info<T, IM>(
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
    pack_vec: fn(*mut IM::Vec, *const T::Vec, usize),
    pack_vec_exceed: fn(*mut IM::Vec, usize),
    pack_zero: fn(T) -> IM,
    vec_cast_back: fn(*const IM::Vec) -> T::Vec,
    cast_back: fn(IM) -> T,
) where
    T: CommonBounds + MatmulMicroKernel + Cast<IM>,
    IM: CommonBounds,
{
    let nr = T::get_max_mixed_precision_nr() * T::Vec::SIZE;
    let mr = T::get_max_mixed_precision_mr().min(m);
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
    if param.mc == 0 {
        param.mc = m.msrv_next_multiple_of(mr);
    }
    if param.nc == 0 {
        param.nc = m.msrv_next_multiple_of(nr);
    }
    matmul_mixed_precision_template::<T, IM>(
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
        pack_vec,
        pack_vec_exceed,
        pack_zero,
        vec_cast_back,
        cast_back,
    );
}

pub(crate) fn f16_matmul<const DEVICE: usize, A>(
    a: &_Tensor<half::f16, Cpu, DEVICE, A>,
    b: &_Tensor<half::f16, Cpu, DEVICE, A>,
    out: Option<_Tensor<half::f16, Cpu, DEVICE, A>>,
    num_threads: usize,
) -> Result<_Tensor<half::f16, Cpu, DEVICE, A>, TensorError>
where
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    type F32Vec = <f32 as TypeCommon>::Vec;
    type F16Vec = <half::f16 as TypeCommon>::Vec;

    let c = matmul_prepare(&a, &b, out)?;
    let m = a.shape()[0] as usize;
    let n = b.shape()[1] as usize;
    let k = a.shape()[1] as usize;

    matmul_mixed_precision_template_no_block_info::<half::f16, f32>(
        a.ptr(),
        b.ptr(),
        c.ptr(),
        m,
        n,
        k,
        a.strides()[a.ndim() - 2],
        b.strides()[b.ndim() - 2],
        c.shape()[c.ndim() - 1] as i64,
        a.strides()[a.ndim() - 1],
        b.strides()[b.ndim() - 1],
        num_threads,
        |packed_b, b, i| unsafe {
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            let b_vec = b.add(i).read_unaligned();
            let val_f32 = b_vec.to_2_f32vec();
            packed_b_vec0.write(val_f32[0]);
            packed_b_vec1.write(val_f32[1]);
        },
        |packed_b, i| unsafe {
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            packed_b_vec0.write(F32Vec::splat(0.0));
            packed_b_vec1.write(F32Vec::splat(0.0));
        },
        |val| val.cast(),
        |val| {
            let vec0 = unsafe { val.read() };
            let vec1 = unsafe { val.add(1).read() };
            F16Vec::from_2_f32vec([vec0, vec1])
        },
        |val| val.cast(),
    );
    Ok(c)
}

pub(crate) fn bf16_matmul<const DEVICE: usize, A>(
    a: &_Tensor<half::bf16, Cpu, DEVICE, A>,
    b: &_Tensor<half::bf16, Cpu, DEVICE, A>,
    out: Option<_Tensor<half::bf16, Cpu, DEVICE, A>>,
    num_threads: usize,
) -> Result<_Tensor<half::bf16, Cpu, DEVICE, A>, TensorError>
where
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    type F32Vec = <f32 as TypeCommon>::Vec;
    type F16Vec = <half::bf16 as TypeCommon>::Vec;

    let c = matmul_prepare(&a, &b, out)?;
    let m = a.shape()[0] as usize;
    let n = b.shape()[1] as usize;
    let k = a.shape()[1] as usize;

    matmul_mixed_precision_template_no_block_info::<half::bf16, f32>(
        a.ptr(),
        b.ptr(),
        c.ptr(),
        m,
        n,
        k,
        a.strides()[a.ndim() - 2],
        b.strides()[b.ndim() - 2],
        c.shape()[c.ndim() - 1] as i64,
        a.strides()[a.ndim() - 1],
        b.strides()[b.ndim() - 1],
        num_threads,
        |packed_b, b, i| unsafe {
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            let b_vec = b.add(i).read_unaligned();
            let val_f32 = b_vec.to_2_f32vec();
            packed_b_vec0.write(val_f32[0]);
            packed_b_vec1.write(val_f32[1]);
        },
        |packed_b, i| unsafe {
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            packed_b_vec0.write(F32Vec::splat(0.0));
            packed_b_vec1.write(F32Vec::splat(0.0));
        },
        |val| val.cast(),
        |val| {
            let vec0 = unsafe { val.read() };
            let vec1 = unsafe { val.add(1).read() };
            F16Vec::from_2_f32vec([vec0, vec1])
        },
        |val| val.cast(),
    );
    Ok(c)
}
