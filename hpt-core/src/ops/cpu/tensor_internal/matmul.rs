use std::borrow::{Borrow, BorrowMut};

use crate::tensor_base::_Tensor;
use crate::THREAD_POOL;
use hpt_common::error::{base::TensorError, shape::ShapeError};
use hpt_common::shape::shape_utils::predict_broadcast_shape;
use hpt_common::shape::shape_utils::{compare_and_pad_shapes, mt_intervals};
use hpt_common::strides::strides_utils::preprocess_strides;
use hpt_traits::TensorLike;
use hpt_traits::{CommonBounds, Matmul, TensorCreator, TensorInfo};
use hpt_types::dtype::TypeCommon;
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn matmul_with_out<A, B, O, Q>(
    lhs: &_Tensor<A>,
    rhs: &_Tensor<B>,
    out: Option<O>,
) -> std::result::Result<_Tensor<<A as NormalOut<B>>::Output>, TensorError>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    O: Borrow<_Tensor<Q>> + BorrowMut<_Tensor<Q>>,
    <A as NormalOut<B>>::Output: CommonBounds,
    Q: CommonBounds,
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let res = if let Some(mut out) = out {
            if out.borrow().size() == ((lhs.shape()[0] * rhs.shape()[1]) as usize)
                && out.borrow().parent().is_none()
            {
                let val = Q::ZERO;
                out.borrow_mut().as_raw_mut().par_iter_mut().for_each(|x| {
                    *x = val;
                });
                let casted: _Tensor<<A as NormalOut<B>>::Output> =
                    out.borrow().static_cast::<<A as NormalOut<B>>::Output>()?;
                casted
            } else {
                _Tensor::<<A as NormalOut<B>>::Output>::zeros(vec![lhs.shape()[0], rhs.shape()[1]])?
            }
        } else {
            _Tensor::<<A as NormalOut<B>>::Output>::zeros(vec![lhs.shape()[0], rhs.shape()[1]])?
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
                <A as NormalOut<B>>::Output::ZERO,
                <A as NormalOut<B>>::Output::ONE,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(rayon::current_num_threads()),
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
            if out.borrow().size() == (res_shape.iter().product::<i64>() as usize) {
                let val = Q::ZERO;
                out.borrow_mut().as_raw_mut().par_iter_mut().for_each(|x| {
                    *x = val;
                });
                out.borrow().static_cast::<<A as NormalOut<B>>::Output>()?
            } else {
                _Tensor::<<A as NormalOut<B>>::Output>::zeros(res_shape)?
            }
        } else {
            _Tensor::<<A as NormalOut<B>>::Output>::zeros(res_shape)?
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
            res_ptr.add((end - start) * res_inner_matrix_size);
            let mut prg: Vec<i64> = vec![0; iterate_shape.len()];
            let mut amount_cpy = amount as i64;
            for j in (0..=iterate_shape.len() - 1).rev() {
                prg[j] = amount_cpy % (iterate_shape[j] + 1);
                amount_cpy /= iterate_shape[j] + 1;
                a_ptr.offset(prg[j] * a_strides[j]);
                b_ptr.offset(prg[j] * b_strides[j]);
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
                                <A as NormalOut<B>>::Output::ZERO,
                                <A as NormalOut<B>>::Output::ONE,
                                false,
                                false,
                                false,
                                gemm::Parallelism::Rayon(threads),
                            );
                            res_ptr.add(res_inner_matrix_size);
                            for j in 0..shape.len() {
                                if prg[j] < shape[j] {
                                    prg[j] += 1;
                                    a_ptr.offset(__a_strides[j]);
                                    b_ptr.offset(__b_strides[j]);
                                    break;
                                } else {
                                    prg[j] = 0;
                                    a_ptr.offset(-__a_strides[j] * shape[j]);
                                    b_ptr.offset(-__b_strides[j] * shape[j]);
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

impl<A, B> Matmul<_Tensor<B>> for _Tensor<A>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
{
    type Output = _Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    fn matmul(&self, rhs: _Tensor<B>) -> Result<Self::Output, TensorError> {
        matmul_with_out(self, &rhs, None::<Self::Output>)
    }
    fn matmul_<U>(&self, rhs: _Tensor<B>, out: U) -> Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(self, &rhs, Some(out))
    }
}

impl<A, B> Matmul<&_Tensor<B>> for _Tensor<A>
where
    A: CommonBounds + NormalOut<B> + Cast<<A as NormalOut<B>>::Output>,
    B: CommonBounds + Cast<<A as NormalOut<B>>::Output>,
    <A as NormalOut<B>>::Output: CommonBounds,
{
    type Output = _Tensor<<A as NormalOut<B>>::Output>;

    type OutputMeta = <A as NormalOut<B>>::Output;

    type InplaceOutput = _Tensor<<A as NormalOut<B>>::Output>;

    fn matmul(&self, rhs: &_Tensor<B>) -> Result<Self::Output, TensorError> {
        matmul_with_out(self, &rhs, None::<Self::Output>)
    }

    fn matmul_<U>(&self, rhs: &_Tensor<B>, out: U) -> Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        matmul_with_out(self, rhs, Some(out))
    }
}
