use std::cmp::min;

use gemm_common::{ cache::CACHE_INFO, gemm::CACHELINE_ALIGN };
use hpt_common::{
    error::{ base::TensorError, shape::ShapeError },
    layout::layout::Layout,
    shape::{ shape::Shape, shape_utils::{ compare_and_pad_shapes, predict_broadcast_shape } },
};
use hpt_traits::tensor::TensorInfo;

use crate::tensor::Tensor;

thread_local! {
    pub(crate) static L2_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(
        dyn_stack::MemBuffer::new(
            dyn_stack::StackReq::new_aligned::<u8>(CACHE_INFO[1].cache_bytes * 8, CACHELINE_ALIGN)
        )
    );
}

thread_local! {
    pub(crate) static L3_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(
        dyn_stack::MemBuffer::new(
            dyn_stack::StackReq::new_aligned::<u8>(
                CACHE_INFO[2].cache_bytes.max(1024 * 1024 * 8) * 8,
                CACHELINE_ALIGN
            )
        )
    );
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
    end: usize
) -> [usize; 3] {
    let mut ret = prg;
    let j_start = prg[0] * nc;
    let mut i_start = prg[1] * mr;
    let mut jj_start = prg[2] * nr;
    for j in (j_start..n).step_by(nc) {
        let jb = min(nc, n - j);
        // pack b
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
    intervals: &[(usize, usize)]
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
#[track_caller]
pub(crate) fn matmul_prepare(
    lhs: &Tensor,
    rhs: &Tensor,
    out: Option<Tensor>
) -> Result<Tensor, TensorError> {
    assert_eq!(lhs.dtype, rhs.dtype);
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let res = if let Some(out) = out {
            assert_eq!(out.dtype, lhs.dtype);
            ShapeError::check_inplace_out_layout_valid(
                &Shape::from([lhs.shape()[0], rhs.shape()[1]]),
                &out.layout()
            )?;
            Ok(out)
        } else {
            Tensor::empty(&[lhs.shape()[0], rhs.shape()[1]], lhs.dtype, lhs.device.clone())
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
        let mut res_shape = predict_broadcast_shape(
            &a_shape[..a_shape.len() - 2],
            &b_shape[..b_shape.len() - 2]
        )?.to_vec();
        res_shape.push(a_shape[a_shape.len() - 2]);
        res_shape.push(b_shape[b_shape.len() - 1]);
        let res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out.layout())?;
            Ok(out)
        } else {
            Tensor::empty(&res_shape, lhs.dtype, lhs.device.clone())
        };
        res
    }
}

pub(crate) fn check_out_layout(
    lhs: &Layout,
    rhs: &Layout,
    out: &Layout,
) -> Result<(), TensorError> {
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        ShapeError::check_inplace_out_layout_valid(
            &Shape::from([lhs.shape()[0], rhs.shape()[1]]),
            &out
        )?;
        Ok(())
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
        let mut res_shape = predict_broadcast_shape(
            &a_shape[..a_shape.len() - 2],
            &b_shape[..b_shape.len() - 2]
        )?.to_vec();
        res_shape.push(a_shape[a_shape.len() - 2]);
        res_shape.push(b_shape[b_shape.len() - 1]);
        ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out)?;
        Ok(())
    }
}
