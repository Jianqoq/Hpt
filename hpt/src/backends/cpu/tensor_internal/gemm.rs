use std::borrow::{Borrow, BorrowMut};

use crate::tensor_base::_Tensor;
use crate::THREAD_POOL;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::{base::TensorError, shape::ShapeError};
use hpt_common::shape::shape::Shape;
use hpt_common::shape::shape_utils::predict_broadcast_shape;
use hpt_common::shape::shape_utils::{compare_and_pad_shapes, mt_intervals};
use hpt_common::strides::strides_utils::preprocess_strides;
use hpt_traits::ops::binary::Gemm;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::shape_manipulate::ShapeManipulate;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};
use num_cpus::get_physical;

type GemmOutput<A, B, const DEVICE: usize, A2> =
    _Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, A2>;

#[track_caller]
pub(crate) fn gemm_with_out<A, B, O, A2, const DEVICE: usize>(
    lhs: &_Tensor<A, Cpu, DEVICE, A2>,
    rhs: &_Tensor<B, Cpu, DEVICE, A2>,
    out: Option<O>,
    alpha: <A as NormalOut<B>>::Output,
    beta: <A as NormalOut<B>>::Output,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
) -> std::result::Result<GemmOutput<A, B, DEVICE, A2>, TensorError>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    O: Borrow<_Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, A2>>
        + BorrowMut<_Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, A2>>,
    <A as NormalOut<B>>::Output: CommonBounds,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let res_shape = vec![lhs.shape()[0], rhs.shape()[1]];
        let res = if let Some(mut out) = out {
            let out: _Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, A2> =
                out.borrow_mut().clone();
            ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out.layout())?;
            out.reshape(&res_shape)?
        } else {
            _Tensor::<<A as NormalOut<B>>::Output, Cpu, DEVICE, A2>::empty(res_shape)?
        };
        let new_a = &lhs.try_astype()?;
        let new_b = &rhs.try_astype()?;
        unsafe {
            gemm::gemm(
                lhs.shape()[0] as usize,
                rhs.shape()[1] as usize,
                rhs.shape()[0] as usize,
                res.data.ptr,
                res.strides()[1] as isize,
                res.strides()[0] as isize,
                false,
                new_a.data.ptr,
                new_a.strides()[1] as isize,
                new_a.strides()[0] as isize,
                new_b.data.ptr,
                new_b.strides()[1] as isize,
                new_b.strides()[0] as isize,
                alpha,
                beta,
                conj_dst,
                conj_lhs,
                conj_rhs,
                gemm::Parallelism::Rayon(rayon::current_num_threads().min(get_physical())),
            );
        }
        Ok(res)
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
        let mut iterate_shape = res_shape.clone();
        res_shape.push(a_shape[a_shape.len() - 2]);
        res_shape.push(b_shape[b_shape.len() - 1]);
        let new_a = &lhs.try_astype::<<A as NormalOut<B>>::Output>()?;
        let new_b = &rhs.try_astype::<<A as NormalOut<B>>::Output>()?;
        let res = if let Some(mut out) = out {
            let out: _Tensor<<A as NormalOut<B>>::Output, Cpu, DEVICE, A2> =
                out.borrow_mut().clone();
            ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out.layout())?;
            out.reshape(&res_shape)?
        } else {
            _Tensor::<<A as NormalOut<B>>::Output, Cpu, DEVICE, A2>::empty(res_shape)?
        };
        let a_strides = preprocess_strides(&a_shape, &lhs.strides());
        let b_strides = preprocess_strides(&b_shape, &rhs.strides());
        let len = iterate_shape.iter().fold(1, |acc, x| acc * (*x as usize));
        let res_inner_matrix_size = (res.shape()[res.shape().len() - 2] as usize)
            * (res.shape()[res.shape().len() - 1] as usize);
        iterate_shape.iter_mut().for_each(|x| {
            *x -= 1;
        });
        let mut a_ptr = new_a.data.clone();
        let mut b_ptr = new_b.data.clone();
        let mut res_ptr = res.data.clone();
        let num_threads = if len < rayon::current_num_threads() {
            len
        } else {
            rayon::current_num_threads()
        };
        let mut num_threads_each: Vec<usize> = if len < rayon::current_num_threads() {
            let vec = mt_intervals(rayon::current_num_threads(), len);
            vec.iter().map(|x| x.1 - x.0).collect::<Vec<usize>>()
        } else {
            vec![1; rayon::current_num_threads()]
        };
        let intervals = mt_intervals(len, num_threads);
        let mut res_ptrs = Vec::with_capacity(num_threads);
        let mut a_ptrs = Vec::with_capacity(num_threads);
        let mut b_ptrs = Vec::with_capacity(num_threads);
        let mut prgs = Vec::with_capacity(num_threads);
        let mut amount = 0;
        for i in 0..num_threads {
            let (start, end) = intervals[i];
            res_ptrs.push(res_ptr.clone());
            res_ptr+=(end - start) * res_inner_matrix_size;
            let mut prg: Vec<i64> = vec![0; iterate_shape.len()];
            let mut amount_cpy = amount as i64;
            for j in (0..=iterate_shape.len() - 1).rev() {
                prg[j] = amount_cpy % (iterate_shape[j] + 1);
                amount_cpy /= iterate_shape[j] + 1;
                a_ptr += prg[j] * a_strides[j];
                b_ptr += prg[j] * b_strides[j];
            }
            amount += end - start;
            a_ptrs.push(a_ptr);
            b_ptrs.push(b_ptr);
            a_ptr = new_a.data.clone();
            b_ptr = new_b.data.clone();
            prgs.push(prg);
        }
        let lhs_cs = lhs.strides()[lhs.strides().len() - 1];
        let lhs_rs = lhs.strides()[lhs.strides().len() - 2];
        let dst_cs = res.strides()[res.strides().len() - 1];
        let dst_rs = res.strides()[res.strides().len() - 2];
        let rhs_cs = rhs.strides()[rhs.strides().len() - 1];
        let rhs_rs = rhs.strides()[rhs.strides().len() - 2];
        let m = a_shape[a_shape.len() - 2] as usize;
        let n = b_shape[b_shape.len() - 1] as usize;
        let k = b_shape[b_shape.len() - 2] as usize;
        THREAD_POOL.with_borrow_mut(|pool: &mut threadpool::ThreadPool| {
            for i in (0..num_threads).rev() {
                let threads: usize = num_threads_each.pop().unwrap();
                let current_size: usize = intervals[i].1 - intervals[i].0;
                let mut res_ptr = res_ptrs.pop().unwrap();
                let mut a_ptr = a_ptrs.pop().unwrap();
                let mut b_ptr = b_ptrs.pop().unwrap();
                let mut prg = prgs.pop().unwrap();
                let shape = iterate_shape.clone();
                let __a_strides = a_strides.clone();
                let __b_strides = b_strides.clone();
                pool.execute(move || {
                    for _ in 0..current_size {
                        unsafe {
                            gemm::gemm(
                                m,
                                n,
                                k,
                                res_ptr.ptr,
                                dst_cs as isize,
                                dst_rs as isize,
                                false,
                                a_ptr.ptr,
                                lhs_cs as isize,
                                lhs_rs as isize,
                                b_ptr.ptr,
                                rhs_cs as isize,
                                rhs_rs as isize,
                                alpha,
                                beta,
                                conj_dst,
                                conj_lhs,
                                conj_rhs,
                                gemm::Parallelism::Rayon(threads),
                            );
                            res_ptr += res_inner_matrix_size;
                            for j in 0..shape.len() {
                                if prg[j] < shape[j] {
                                    prg[j] += 1;
                                    a_ptr += __a_strides[j];
                                    b_ptr += __b_strides[j];
                                    break;
                                } else {
                                    prg[j] = 0;
                                    a_ptr += -__a_strides[j] * shape[j];
                                    b_ptr += -__b_strides[j] * shape[j];
                                }
                            }
                        }
                    }
                });
            }
            pool.join();
        });
        Ok(res)
    }
}

impl<A, B, A2, const DEVICE: usize> Gemm<_Tensor<B, Cpu, DEVICE, A2>>
    for _Tensor<A, Cpu, DEVICE, A2>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = GemmOutput<A, B, DEVICE, A2>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = GemmOutput<A, B, DEVICE, A2>;

    fn gemm(
        &self,
        rhs: _Tensor<B, Cpu, DEVICE, A2>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
    ) -> Result<Self::Output, TensorError> {
        gemm_with_out(
            self,
            &rhs,
            None::<Self::Output>,
            alpha,
            beta,
            conj_dst,
            conj_lhs,
            conj_rhs,
        )
    }
    fn gemm_<U>(
        &self,
        rhs: _Tensor<B, Cpu, DEVICE, A2>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
        out: U,
    ) -> Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        gemm_with_out(
            self,
            &rhs,
            Some(out),
            alpha,
            beta,
            conj_dst,
            conj_lhs,
            conj_rhs,
        )
    }
}

impl<A, B, A2, const DEVICE: usize> Gemm<&_Tensor<B, Cpu, DEVICE, A2>>
    for _Tensor<A, Cpu, DEVICE, A2>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    type Output = GemmOutput<A, B, DEVICE, A2>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = GemmOutput<A, B, DEVICE, A2>;

    fn gemm(
        &self,
        rhs: &_Tensor<B, Cpu, DEVICE, A2>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
    ) -> Result<Self::Output, TensorError> {
        gemm_with_out(
            self,
            &rhs,
            None::<Self::Output>,
            alpha,
            beta,
            conj_dst,
            conj_lhs,
            conj_rhs,
        )
    }

    fn gemm_<U>(
        &self,
        rhs: &_Tensor<B, Cpu, DEVICE, A2>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
        out: U,
    ) -> Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        gemm_with_out(
            self,
            rhs,
            Some(out),
            alpha,
            beta,
            conj_dst,
            conj_lhs,
            conj_rhs,
        )
    }
}
