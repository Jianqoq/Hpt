use std::cmp::min;

use gemm_common::{cache::CACHE_INFO, gemm::CACHELINE_ALIGN};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_common::{
    error::{base::TensorError, shape::ShapeError},
    shape::{
        shape::Shape,
        shape_utils::{compare_and_pad_shapes, predict_broadcast_shape},
    },
    Pointer,
};
use hpt_traits::{
    ops::creation::TensorCreator,
    tensor::{CommonBounds, TensorInfo},
};
use hpt_types::into_scalar::Cast;

use crate::tensor_base::_Tensor;
use hpt_types::traits::VecTrait;

thread_local! {
    pub(crate) static L2_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(dyn_stack::MemBuffer::new(
        dyn_stack::StackReq::new_aligned::<u8>(CACHE_INFO[1].cache_bytes, CACHELINE_ALIGN)
    ));
}

thread_local! {
    pub(crate) static L3_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(dyn_stack::MemBuffer::new(
        dyn_stack::StackReq::new_aligned::<u8>(CACHE_INFO[2].cache_bytes.max(1024 * 1024 * 32), CACHELINE_ALIGN)
    ));
}

pub(crate) fn calculate_jobs(n: usize, nc: usize, mr: usize, nr: usize, ib: usize) -> usize {
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

pub(crate) fn calculate_prg(
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

pub(crate) fn calculate_prgs(
    n: usize,
    nc: usize,
    mr: usize,
    nr: usize,
    ib: usize,
    intervals: &[(usize, usize)],
) -> Vec<[usize; 3]> {
    let mut prgs = vec![[0, 0, 0]; intervals.len()];
    let mut prg = [0, 0, 0];
    for (tid, (start, end)) in intervals.iter().enumerate() {
        prgs[tid] = prg;
        prg = calculate_prg(n, nc, mr, nr, ib, prg, *start, *end);
    }
    prgs
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
) where
    T: CommonBounds + Cast<I>,
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
                let row = i as i64 + ii;
                *packed_a = a[row * lda + p * stride].cast();
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
    pack_zero: fn(T) -> I,
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
                        unsafe { b.ptr.offset((p * ldb) as isize + j as isize) } as *const T::Vec,
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
                    let j = j as i64 + jj;
                    *packed_b = pack_zero(b[p * ldb + j * stride]);
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

#[inline(always)]
pub(crate) fn matmul_prepare<T, const DEVICE: usize, A>(
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
