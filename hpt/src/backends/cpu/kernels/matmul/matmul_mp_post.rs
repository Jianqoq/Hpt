use std::cmp::min;

use crate::{
    backends::cpu::kernels::matmul::common::{
        calculate_jobs, calculate_prgs, matmul_prepare, pack_a_mixed_precision,
        pack_b_mixed_precision, L2_SLAB,
    },
    tensor_base::_Tensor,
    ALIGN, RAYON_POOL,
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
use hpt_types::{dtype::TypeCommon, traits::VecTrait};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use super::{microkernel_trait::MatmulMicroKernel, utils::kernel_params};

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
pub fn matmul_mixed_precision_template<T, IM, F1, F2>(
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
    pack_zero: fn(&mut IM, &T),
    vec_cast_back: fn(*mut T::Vec, *const IM::Vec),
    cast_back: fn(&mut T, &IM),
    post_op: F1,
    post_op_vec: F2,
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
    let packed_a_layout = std::alloc::Layout::from_size_align(
        num_mr_blocks * mr * kc * std::mem::size_of::<IM>(),
        ALIGN,
    )
    .expect("layout create failed");

    let packed_a = {
        let a_buffer = unsafe { std::alloc::alloc(packed_a_layout) };
        let ret = Pointer::new(
            a_buffer as *mut IM,
            (packed_a_layout.size() / std::mem::size_of::<IM>()) as i64,
        );
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
    RAYON_POOL.with_borrow(|pool| {
        pool.install(|| {
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
                            let packed_b = Pointer::new(
                                packed_b_storage.as_mut_ptr() as *mut IM,
                                (num_nr_blocks * nr * kc) as i64,
                            );
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
                                    let last_kiter = p + pb == k;
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
                                        pack_zero,
                                    );
                                    barrier.wait();

                                    let mut job_count = use_start;
                                    let mut i_start = use_prg[1] * mr;
                                    let mut jj_start = use_prg[2] * nr;
                                    'outer: for j in (j_start..n).step_by(nc) {
                                        let jb = min(nc, n - j);
                                        let c = out.clone() + i as i64 * ldc + j as i64;
                                        pack_b_mixed_precision::<T, IM>(
                                            b.clone()
                                                + (p as i64 * ldb + j as i64 * rhs_col_stride),
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
                                        for ii in (i_start..ib).step_by(mr) {
                                            let mb = min(mr, ib - ii);
                                            let micro_kernel =
                                                <T>::get_mixed_precision_kernel_with_post_op(
                                                    nr / <T>::Vec::SIZE,
                                                    mb,
                                                );

                                            for jj in (jj_start..jb).step_by(nr) {
                                                let jjb = min(nr, jb - jj);
                                                micro_kernel(
                                                    packed_a.clone() + kc as i64 * ii as i64,
                                                    packed_b.clone() + jj as i64 * kc as i64,
                                                    c.clone() + ii as i64 * ldc + jj as i64,
                                                    ldc,
                                                    1,
                                                    kc,
                                                    jjb,
                                                    mb as i64,
                                                    first_kiter,
                                                    last_kiter,
                                                    i + ii,
                                                    j + jj,
                                                    vec_cast_back,
                                                    cast_back,
                                                    post_op.clone(),
                                                    post_op_vec.clone(),
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
        });
    });

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
pub fn matmul_mp_post_template_no_block_info<T, IM, F1, F2>(
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
    pack_zero: fn(&mut IM, &T),
    vec_cast_back: fn(*mut T::Vec, *const IM::Vec),
    cast_back: fn(&mut T, &IM),
    post_op: F1,
    post_op_vec: F2,
) where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    let nr = T::get_max_mixed_precision_nr() * T::Vec::SIZE;
    let mr = T::get_max_mixed_precision_mr();
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
        kernel_params(n, m, k, nr, mr, std::mem::size_of::<T>(), true)
    };
    if param.mc == 0 {
        param.mc = m.msrv_next_multiple_of(mr);
    }
    if param.nc == 0 {
        param.nc = n.msrv_next_multiple_of(nr);
    }
    matmul_mixed_precision_template::<T, IM, F1, F2>(
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
        num_threads,
        pack_vec,
        pack_vec_exceed,
        pack_zero,
        vec_cast_back,
        cast_back,
        post_op,
        post_op_vec,
    );
}

#[duplicate::duplicate_item(
    func_name half_type half_str;
    [f16_matmul_mp_post_template_no_block_info] [half::f16] ["f16"];
    [bf16_matmul_mp_post_template_no_block_info] [half::bf16] ["bf16"];
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
    num_threads: usize,
    post_op: F1,
    post_op_vec: F2,
) where
    T: CommonBounds + MatmulMicroKernel,
    IM: CommonBounds,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(T::Vec, usize, usize) -> T::Vec + Clone + Send + Sync + 'static,
{
    type F32Vec = <f32 as TypeCommon>::Vec;
    type F16Vec = <half_type as TypeCommon>::Vec;
    assert_eq!(T::STR, half_str);
    assert_eq!(IM::STR, "f32");
    matmul_mp_post_template_no_block_info::<T, f32, F1, F2>(
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
        |packed_b, b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let b = b as *const F16Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            let b_vec = b.add(i).read_unaligned();
            let val_f32 = b_vec.to_2_f32vec();
            packed_b_vec0.write(val_f32[0]);
            packed_b_vec1.write(val_f32[1]);
        },
        |packed_b, i| unsafe {
            let packed_b = packed_b as *mut F32Vec;
            let packed_b_vec0 = packed_b.add(i * 2);
            let packed_b_vec1 = packed_b.add(i * 2 + 1);
            packed_b_vec0.write(F32Vec::splat(0.0));
            packed_b_vec1.write(F32Vec::splat(0.0));
        },
        |im, val| {
            let val = val as *const T as *const half_type;
            *im = unsafe { val.read().to_f32() };
        },
        |im, val| {
            let im = im as *mut F16Vec;
            let val = val as *const F32Vec;
            let vec0 = unsafe { val.read() };
            let vec1 = unsafe { val.add(1).read() };
            unsafe { im.write_unaligned(F16Vec::from_2_f32vec([vec0, vec1])) };
        },
        |im, val| {
            let im = im as *mut T as *mut half_type;
            unsafe { *im = half_type::from_f32(*val) };
        },
        post_op,
        post_op_vec,
    );
}

#[duplicate::duplicate_item(
    func_name inner_func_name half_type;
    [f16_matmul_post] [f16_matmul_mp_post_template_no_block_info] [half::f16];
    [bf16_matmul_post] [bf16_matmul_mp_post_template_no_block_info] [half::bf16];
)]
#[allow(unused)]
pub(crate) fn func_name<T: CommonBounds, const DEVICE: usize, A, F1, F2>(
    a: &_Tensor<T, Cpu, DEVICE, A>,
    b: &_Tensor<T, Cpu, DEVICE, A>,
    out: Option<_Tensor<T, Cpu, DEVICE, A>>,
    post_op: F1,
    post_op_vec: F2,
    num_threads: usize,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: MatmulMicroKernel,
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(<T as TypeCommon>::Vec, usize, usize) -> <T as TypeCommon>::Vec
        + Clone
        + Send
        + Sync
        + 'static,
{
    let c = matmul_prepare(&a, &b, out)?;
    let m = a.shape()[0] as usize;
    let n = b.shape()[1] as usize;
    let k = a.shape()[1] as usize;

    inner_func_name::<T, f32, F1, F2>(
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
        post_op,
        post_op_vec,
    );
    Ok(c)
}
