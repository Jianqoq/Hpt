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
use hpt_common::shape::shape_utils::compare_and_pad_shapes;
use hpt_common::shape::shape_utils::predict_broadcast_shape;
use hpt_common::{error::base::TensorError, Pointer};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::traits::VecTrait;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
            _Tensor::<T, Cpu, DEVICE, A>::zeros(vec![lhs.shape()[0], rhs.shape()[1]])
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
            _Tensor::<T, Cpu, DEVICE, A>::zeros(res_shape)
        };
        res
    }
}

thread_local! {
    pub static L2_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(dyn_stack::MemBuffer::new(
        dyn_stack::StackReq::new_aligned::<u8>(CACHE_INFO[1].cache_bytes, CACHELINE_ALIGN)
    ));
}

pub(crate) fn gemm2d<T>(
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
    _nc: usize,
    nr: usize,
    mr: usize,
) where
    T: CommonBounds + MicroKernel,
{
    assert_eq!(
        nr % T::Vec::SIZE,
        0,
        "nr must be a multiple of {} for type {}",
        T::Vec::SIZE,
        T::STR
    );

    let mut do_lhs_pack = false;

    // in microkernel, lhs_load: mr * kc, rhs_load: nr * kc, dst_load: mr * nr
    // only pack lhs if TLB is not enough or lhs_col_stride is not 1
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

    let n_blocks = n.div_ceil(nr);
    let num_threads = n_blocks.min(rayon::current_num_threads());
    let blocks_per_thread = n_blocks.div_ceil(num_threads);
    let num_mr_blocks = (mc + mr - 1) / mr;
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

    let pack_a_fn = if do_lhs_pack {
        #[inline(always)]
        fn pack<T>(
            a: Pointer<T>,
            packed_a: Pointer<T>,
            lda: i64,
            stride: i64,
            kc: usize,
            mr: usize,
            i: usize,
            p: usize,
            ib: usize,
            pb: usize,
            tid: usize,
            mb_per_thread: usize,
            num_mr_blocks: usize,
        ) where
            T: CommonBounds,
        {
            pack_a::<T>(
                a + i as i64 * lda + p as i64 * stride,
                packed_a,
                lda,
                stride,
                ib,
                pb,
                kc,
                mr,
                tid,
                mb_per_thread,
                num_mr_blocks,
            );
        }
        pack::<T>
    } else {
        #[allow(unused)]
        #[inline(always)]
        fn pack<T>(
            a: Pointer<T>,
            packed_a: Pointer<T>,
            lda: i64,
            stride: i64,
            kc: usize,
            i: usize,
            p: usize,
            ib: usize,
            pb: usize,
            mr: usize,
            tid: usize,
            mb_per_thread: usize,
            num_mr_blocks: usize,
        ) where
            T: CommonBounds,
        {
        }
        pack::<T>
    };

    let packed_a_ptr = packed_a.ptr as *mut T;

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(num_threads));

    (0..num_threads).into_par_iter().for_each(|tid| {
        L2_SLAB.with(|mem| {
            let mut mem = mem.borrow_mut();
            let stack = DynStack::new(&mut mem);
            let (packed_b_storage, _) = stack.make_aligned_uninit::<T>(nr * kc, ALIGN);
            #[cfg(feature = "bound_check")]
            let packed_b = Pointer::new(packed_b_storage.as_mut_ptr() as *mut T, (nr * kc) as i64);
            #[cfg(not(feature = "bound_check"))]
            let packed_b = Pointer::new(packed_b_storage.as_mut_ptr() as *mut T);

            for i in (0..m).step_by(mc) {
                let ib = min(mc, m - i);
                for p in (0..k).step_by(kc) {
                    let pb = min(kc, k - p);

                    pack_a_fn(
                        a.clone(),
                        packed_a.clone(),
                        lda,
                        lhs_col_stride,
                        kc,
                        mr,
                        i,
                        p,
                        ib,
                        pb,
                        tid,
                        mb_per_thread,
                        num_mr_blocks,
                    );

                    barrier.wait();

                    let start_block = tid * blocks_per_thread;
                    let end_block = min((tid + 1) * blocks_per_thread, n_blocks);
                    for block in start_block..end_block {
                        let j = block * nr;
                        let jb = min(nr, n - j);
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
                        outer_kernel::<T>(
                            if do_lhs_pack {
                                packed_a.clone()
                            } else {
                                a.clone() + (i as i64 * lda + p as i64 * lhs_col_stride)
                            },
                            packed_b.clone(),
                            out.clone() + i as i64 * ldc + j as i64,
                            ib,
                            jb,
                            mr,
                            nr,
                            ldc,
                            lda,
                            kc,
                            do_lhs_pack,
                            p == 0,
                        );
                    }

                    barrier.wait();
                }
            }
        });
    });

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
                        packed_b_vec.write_unaligned(
                            (b.ptr
                                .offset((p * ldb) as isize + (i * T::Vec::SIZE) as isize)
                                as *const T::Vec)
                                .read_unaligned(),
                        )
                    };
                }
                packed_b += nb as i64;
            }
            for _ in kb..kc {
                for _ in 0..nb as i64 {
                    for i in 0..nr_div_lane {
                        let packed_b_vec =
                            unsafe { packed_b.ptr.add(i * T::Vec::SIZE) } as *mut T::Vec;
                        unsafe { packed_b_vec.write_unaligned(T::Vec::splat(T::ZERO)) };
                    }
                    packed_b += nb as i64;
                }
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

#[inline(never)]
pub(crate) fn outer_kernel<T>(
    packed_a: Pointer<T>,
    mut packed_b: Pointer<T>,
    c: Pointer<T>,
    mc: usize,
    nc: usize,
    mr: usize,
    nr: usize,
    ldc: i64,
    lda: i64,
    kc: usize,
    do_lhs_pack: bool,
    first_kiter: bool,
) where
    T: CommonBounds + MicroKernel,
{
    let packed_b_cpy = packed_b.clone();
    for (idx, i) in (0..mc).step_by(mr).enumerate() {
        let ib = min(mr, mc - i);
        packed_b = packed_b_cpy.clone();
        let micro_kernel = T::get_kernel(nr / T::Vec::SIZE, ib);
        if do_lhs_pack {
            micro_kernel(
                packed_a.clone() + kc as i64 * mr as i64 * idx as i64,
                packed_b.clone(),
                c.clone() + i as i64 * ldc,
                ldc,
                1,
                kc,
                nc,
                ib as i64,
                first_kiter,
            );
        } else {
            micro_kernel(
                packed_a.clone() + i as i64 * lda,
                packed_b.clone(),
                c.clone() + i as i64 * ldc,
                ldc,
                lda,
                kc,
                nc,
                1,
                first_kiter,
            );
        }
        packed_b += nr * kc;
    }
}

/// gemm
pub fn gemm<T, const DEVICE: usize, A>(
    a: &Tensor<T, Cpu, DEVICE, A>,
    b: &Tensor<T, Cpu, DEVICE, A>,
    out: Option<Tensor<T, Cpu, DEVICE, A>>,
) -> Result<Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: CommonBounds + MicroKernel,
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
    gemm2d::<T>(
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
        16,
        T::get_max_mr(),
    );
    Ok(c.into())
}
