#![allow(unused)]
use std::cmp::min;

use dyn_stack::DynStack;
use gemm_common::cache::CACHE_INFO;
use gemm_common::gemm::CACHELINE_ALIGN;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape::Shape;
use hpt_common::shape::shape_utils::predict_broadcast_shape;
use hpt_common::{error::base::TensorError, Pointer};
use hpt_traits::ops::creation::TensorCreator;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::backend::Cpu;
use crate::tensor_base::_Tensor;
use crate::{Tensor, ALIGN};
use hpt_common::shape::shape_utils::compare_and_pad_shapes;
use hpt_traits::tensor::{CommonBounds, TensorInfo};

use super::microkernel_trait::MicroKernel;

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

pub(crate) fn gemm2d<T, const MR: usize, const NR: usize, const NR_DIV_LANE: usize>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    stride: i64,
    kc: usize,
    mc: usize,
    nc: usize,
) where
    T: CommonBounds + MicroKernel,
{
    // let num_mr_blocks = (mc + MR - 1) / MR;
    // let packed_a_layout = std::alloc::Layout::from_size_align(
    //     num_mr_blocks * MR * kc * std::mem::size_of::<T>(),
    //     ALIGN,
    // )
    // .expect("layout create failed");
    let n_blocks = n.div_ceil(NR);
    let num_threads = n_blocks.min(rayon::current_num_threads());
    let blocks_per_thread = n_blocks.div_ceil(num_threads);
    // let packed_a_origin = unsafe { std::alloc::alloc(packed_a_layout) };
    // if packed_a_origin == std::ptr::null_mut() {
    //     panic!("alloc failed");
    // }
    // #[cfg(feature = "bound_check")]
    // let packed_a = Pointer::new(
    //     packed_a_origin as *mut T,
    //     (packed_a_layout.size() / std::mem::size_of::<T>()) as i64,
    // );
    // #[cfg(not(feature = "bound_check"))]
    // let packed_a = Pointer::new(packed_a_origin as *mut T);

    for i in (0..m).step_by(mc) {
        let ib = min(mc, m - i);
        for p in (0..k).step_by(kc) {
            let pb = min(kc, k - p);
            // pack_a::<T, MR>(
            //     a.clone() + i as i64 * lda + p as i64,
            //     packed_a.clone(),
            //     lda,
            //     stride,
            //     ib,
            //     pb,
            //     kc,
            // );
            (0..num_threads).into_par_iter().for_each(|tid| {
                L2_SLAB.with(|mem| {
                    let mut mem = mem.borrow_mut();
                    let stack = DynStack::new(&mut mem);
                    let (packed_b_storage, _) = stack.make_aligned_uninit::<T>(NR * kc, ALIGN);
                    #[cfg(feature = "bound_check")]
                    let packed_b =
                        Pointer::new(packed_b_storage.as_mut_ptr() as *mut T, (NR * kc) as i64);
                    #[cfg(not(feature = "bound_check"))]
                    let packed_b = Pointer::new(packed_b_storage.as_mut_ptr() as *mut T);
                    let start_block = tid * blocks_per_thread;
                    let end_block = min((tid + 1) * blocks_per_thread, n_blocks);
                    for block in start_block..end_block {
                        let j = block * NR;
                        let jb = min(NR, n - j);
                        pack_b::<T, NR>(
                            b.clone() + (p as i64 * ldb + j as i64),
                            packed_b.clone(),
                            ldb,
                            jb,
                            pb,
                            kc,
                        );
                        outer_kernel::<T, MR, NR, NR_DIV_LANE>(
                            a.clone() + i as i64 * lda + p as i64 * stride,
                            packed_b.clone(),
                            out.clone() + i as i64 * ldc + j as i64,
                            ib,
                            jb,
                            ldc,
                            lda,
                            kc,
                        );
                    }
                });
            });
        }
    }
    // unsafe {
    //     std::alloc::dealloc(packed_a.ptr as *mut u8, packed_a_layout);
    // }
}

pub(crate) fn pack_a<T, const MR: usize>(
    a: Pointer<T>,
    mut packed_a: Pointer<T>,
    lda: i64,
    stride: i64,
    mc: usize,
    kb: usize,
    kc: usize,
) where
    T: CommonBounds,
{
    for i in (0..mc).step_by(MR) {
        let mr = MR.min(mc - i);
        for p in 0..kb as i64 {
            for ii in 0..mr as i64 {
                let i = i as i64 + ii;
                *packed_a = a[i * lda + p * stride];
                packed_a += 1i64;
            }
            for _ in mr..MR {
                *packed_a = T::ZERO;
                packed_a += 1i64;
            }
        }
        for _ in kb..kc {
            for _ in 0..mr as i64 {
                *packed_a = T::ZERO;
                packed_a += 1i64;
            }
            for _ in mr..MR {
                *packed_a = T::ZERO;
                packed_a += 1i64;
            }
        }
    }
}

pub(crate) fn pack_b<T, const NR: usize>(
    b: Pointer<T>,
    mut packed_b: Pointer<T>,
    ldb: i64,
    nc: usize,
    kb: usize,
    kc: usize,
) where
    T: CommonBounds,
{
    for j in (0..nc).step_by(NR) {
        let nr = NR.min(nc - j);
        if nr == NR {
            for p in 0..kb as i64 {
                let packed_b_vec = packed_b.ptr as *mut [T; NR];
                unsafe {
                    *packed_b_vec = (b.ptr.offset((p * ldb) as isize) as *const [T; NR]).read()
                };
                packed_b += NR as i64;
            }
            for _ in kb..kc {
                for _ in 0..nr as i64 {
                    let packed_b_vec = packed_b.ptr as *mut [T; NR];
                    unsafe { *packed_b_vec = [T::ZERO; NR] };
                    packed_b += NR as i64;
                }
            }
        } else {
            for p in 0..kb as i64 {
                for jj in 0..nr as i64 {
                    let j = j as i64 + jj;
                    *packed_b = b[p * ldb + j];
                    packed_b += 1i64;
                }
                for _ in nr..NR {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
            }
            for _ in kb..kc {
                for _ in 0..nr as i64 {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
                for _ in nr..NR {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
            }
        }
    }
}

#[inline(never)]
pub(crate) fn outer_kernel<T, const MR: usize, const NR: usize, const NR_DIV_LANE: usize>(
    packed_a: Pointer<T>,
    mut packed_b: Pointer<T>,
    c: Pointer<T>,
    mc: usize,
    nc: usize,
    ldc: i64,
    lda: i64,
    kc: usize,
) where
    T: CommonBounds + MicroKernel,
{
    let packed_b_cpy = packed_b.clone();
    for i in (0..mc).step_by(MR) {
        let ib = min(MR, mc - i);
        packed_b = packed_b_cpy.clone();
        let micro_kernel = T::get_kernel(NR_DIV_LANE, ib);
        for j in (0..nc).step_by(NR) {
            let jb = min(NR, nc - j);
            micro_kernel(
                packed_a.clone() + i as i64 * lda,
                packed_b.clone(),
                c.clone() + i as i64 * ldc + j as i64,
                ldc,
                lda,
                kc,
                jb,
            );
            packed_b += NR * kc;
        }
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
    let lda = a.shape()[1] as i64;
    let ldb = b.shape()[1] as i64;
    let ldc = c.shape()[1] as i64;
    let stride = 1;

    let param = gemm_common::cache::kernel_params(n, m, k, 16, 6, std::mem::size_of::<T>());
    gemm2d::<T, 6, 16, { 16 / 8 }>(
        a.ptr(),
        b.ptr(),
        c.ptr(),
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        stride,
        param.kc,
        param.nc,
        param.mc,
    );
    Ok(c.into())
}
