use rayon::iter::{ IntoParallelRefMutIterator, ParallelIterator };
use std::panic::Location;
use std::sync::{ Arc, Barrier };
use tensor_traits::tensor::CommonBounds;
use tensor_types::type_promote::NormalOut;
use tensor_types::into_scalar::IntoScalar;
use tensor_common::shape_utils::compare_and_pad_shapes;
use tensor_common::shape_utils::predict_broadcast_shape;
use tensor_common::strides_utils::preprocess_strides;
use tensor_common::pointer::Pointer;
use tensor_common::shape_utils::mt_intervals;
use crate::tensor_base::_Tensor;
use crate::THREAD_POOL;
use tensor_traits::tensor::TensorLike;
use tensor_traits::tensor::TensorInfo;
use tensor_traits::tensor::TensorCreator;
use tensor_types::dtype::TypeCommon;
use crate::backend::Cpu;

#[doc = r" Performs matrix multiplication without an output tensor."]
#[doc = r""]
#[doc = r" Given two tensors, this function multiplies them following the rules of matrix multiplication."]
#[doc = r" It automatically handles broadcasting of shapes, if necessary."]
#[doc = r""]
#[doc = r" # Arguments"]
#[doc = r" - `lhs`: Reference to the left-hand side tensor."]
#[doc = r" - `rhs`: Reference to the right-hand side tensor."]
#[doc = r""]
#[doc = r" # Returns"]
#[doc = r" A `Result` containing either the product tensor or an error."]
#[doc = r""]
#[doc = r" # Errors"]
#[doc = r" Returns an error if shapes are not compatible for matrix multiplication."]
#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn matmul_no_out<A, B>(
    lhs: &_Tensor<A>,
    rhs: &_Tensor<B>
)
    -> anyhow::Result<_Tensor<<A as NormalOut<B>>::Output>>
    where
        A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
        B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
        <A as NormalOut<B>>::Output: CommonBounds
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        if lhs.shape()[1] != rhs.shape()[0] {
            anyhow::bail!(
                "shape mismatch when trying to perform matrix multiplication got {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            );
        } else {
            let res;
            res = _Tensor::<<A as NormalOut<B>>::Output, Cpu>::zeros(
                vec![lhs.shape()[0], rhs.shape()[1]]
            )?;
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
                    gemm::Parallelism::Rayon(rayon::current_num_threads())
                );
            }
            Ok(res)
        }
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
        if a_shape[a_shape.len() - 1] != b_shape[b_shape.len() - 2] {
            anyhow::bail!(
                "shape mismatch when trying to perform matrix multiplication got {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            );
        } else {
            let mut res_shape = predict_broadcast_shape(
                &a_shape[..a_shape.len() - 2],
                &b_shape[..b_shape.len() - 2],
                Location::caller(),
            )?.to_vec();
            let mut iterate_shape: Vec<i64> = res_shape.clone();
            res_shape.push(a_shape[a_shape.len() - 2]);
            res_shape.push(b_shape[b_shape.len() - 1]);
            let new_a = &lhs.try_astype::<<A as NormalOut<B>>::Output>()?;
            let new_b = &rhs.try_astype::<<A as NormalOut<B>>::Output>()?;
            let res;
            res = _Tensor::<<A as NormalOut<B>>::Output>::zeros(res_shape)?;
            let a_strides: Vec<i64> = preprocess_strides(&a_shape, &lhs.strides());
            let b_strides: Vec<i64> = preprocess_strides(&b_shape, &rhs.strides());
            let len: usize = iterate_shape.iter().fold(1, |acc, x| acc * (*x as usize));
            let res_inner_matrix_size: usize =
                (res.shape()[res.shape().len() - 2] as usize) *
                (res.shape()[res.shape().len() - 1] as usize);
            iterate_shape.iter_mut().for_each(|x| {
                *x -= 1;
            });
            let mut a_ptr: Pointer<<A as NormalOut<B>>::Output> = new_a.data;
            let mut b_ptr: Pointer<<A as NormalOut<B>>::Output> = new_b.data;
            let mut res_ptr: Pointer<<A as NormalOut<B>>::Output> = res.data;
            let num_threads: usize = if len < rayon::current_num_threads() {
                len
            } else {
                rayon::current_num_threads()
            };
            let mut num_threads_each = if len < rayon::current_num_threads() {
                let vec = mt_intervals(rayon::current_num_threads(), len);
                vec.iter()
                    .map(|x| x.1 - x.0)
                    .collect::<Vec<usize>>()
            } else {
                vec![1;
                rayon::current_num_threads()]
            };
            let intervals = mt_intervals(len, num_threads);
            let mut res_ptrs = Vec::with_capacity(num_threads);
            let mut a_ptrs = Vec::with_capacity(num_threads);
            let mut b_ptrs = Vec::with_capacity(num_threads);
            let mut prgs = Vec::with_capacity(num_threads);
            let mut amount = 0;
            for i in 0..num_threads {
                let (start, end) = intervals[i];
                res_ptrs.push(res_ptr);
                res_ptr.add((end - start) * res_inner_matrix_size);
                let mut prg = vec![0;
                iterate_shape.len()];
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
                a_ptr = new_a.data;
                b_ptr = new_b.data;
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
                let barrier = Arc::new(Barrier::new(num_threads + 1));
                for i in (0..num_threads).rev() {
                    let threads = num_threads_each.pop().unwrap();
                    let current_size = intervals[i].1 - intervals[i].0;
                    let mut res_ptr = res_ptrs.pop().unwrap();
                    let mut a_ptr = a_ptrs.pop().unwrap();
                    let mut b_ptr = b_ptrs.pop().unwrap();
                    let mut prg = prgs.pop().unwrap();
                    let shape = iterate_shape.clone();
                    let __a_strides = a_strides.clone();
                    let __b_strides = b_strides.clone();
                    let barrier_clone = Arc::clone(&barrier);
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
                                    gemm::Parallelism::Rayon(threads)
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
                        barrier_clone.wait();
                    });
                }
                barrier.wait();
            });
            Ok(res)
        }
    }
}

#[track_caller]
pub(crate) fn matmul_with_out<A, B, O>(
    lhs: &_Tensor<A>,
    rhs: &_Tensor<B>,
    mut out: O
)
    -> anyhow::Result<_Tensor<<A as NormalOut<B>>::Output>>
    where
        A: CommonBounds + NormalOut<B> + IntoScalar<<A as NormalOut<B>>::Output>,
        B: CommonBounds + IntoScalar<<A as NormalOut<B>>::Output>,
        O: TensorLike<<A as NormalOut<B>>::Output, Output = _Tensor<<A as NormalOut<B>>::Output>> +
            TensorInfo<<A as NormalOut<B>>::Output>,
        <A as NormalOut<B>>::Output: CommonBounds
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        if lhs.shape()[1] != rhs.shape()[0] {
            anyhow::bail!(
                "shape mismatch when trying to perform matrix multiplication got {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            );
        } else {
            let res;
            if out.size() == ((lhs.shape()[0] * rhs.shape()[1]) as usize) && out.parent().is_none() {
                let val = <A as NormalOut<B>>::Output::ZERO;
                out.to_raw_mut()
                    .par_iter_mut()
                    .for_each(|x| {
                        *x = val;
                    });
                res = out.static_cast()?;
            } else {
                res = _Tensor::zeros(vec![lhs.shape()[0], rhs.shape()[1]])?;
            }
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
                    gemm::Parallelism::Rayon(rayon::current_num_threads())
                );
            }
            Ok(res)
        }
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
        if a_shape[a_shape.len() - 1] != b_shape[b_shape.len() - 2] {
            anyhow::bail!(
                "shape mismatch when trying to perform matrix multiplication got {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            );
        } else {
            let mut res_shape = predict_broadcast_shape(
                &a_shape[..a_shape.len() - 2],
                &b_shape[..b_shape.len() - 2],
                Location::caller(),
            )?.to_vec();
            let mut iterate_shape = res_shape.clone();
            res_shape.push(a_shape[a_shape.len() - 2]);
            res_shape.push(b_shape[b_shape.len() - 1]);
            let new_a = &lhs.try_astype::<<A as NormalOut<B>>::Output>()?;
            let new_b = &rhs.try_astype::<<A as NormalOut<B>>::Output>()?;
            let res;
            if out.size() == (res_shape.iter().product::<i64>() as usize) {
                let val = <A as NormalOut<B>>::Output::ZERO;
                out.to_raw_mut()
                    .par_iter_mut()
                    .for_each(|x| {
                        *x = val;
                    });
                res = out.static_cast()?;
            } else {
                res = _Tensor::zeros(res_shape)?;
            }
            let a_strides = preprocess_strides(&a_shape, &lhs.strides());
            let b_strides = preprocess_strides(&b_shape, &rhs.strides());
            let len = iterate_shape.iter().fold(1, |acc, x| acc * (*x as usize));
            let res_inner_matrix_size =
                (res.shape()[res.shape().len() - 2] as usize) *
                (res.shape()[res.shape().len() - 1] as usize);
            iterate_shape.iter_mut().for_each(|x| {
                *x -= 1;
            });
            let mut a_ptr = new_a.data;
            let mut b_ptr = new_b.data;
            let mut res_ptr = res.data;
            let num_threads = if len < rayon::current_num_threads() {
                len
            } else {
                rayon::current_num_threads()
            };
            let mut num_threads_each: Vec<usize> = if len < rayon::current_num_threads() {
                let vec = mt_intervals(rayon::current_num_threads(), len);
                vec.iter()
                    .map(|x| x.1 - x.0)
                    .collect::<Vec<usize>>()
            } else {
                vec![1;
                rayon::current_num_threads()]
            };
            let intervals = mt_intervals(len, num_threads);
            let mut res_ptrs = Vec::with_capacity(num_threads);
            let mut a_ptrs = Vec::with_capacity(num_threads);
            let mut b_ptrs = Vec::with_capacity(num_threads);
            let mut prgs = Vec::with_capacity(num_threads);
            let mut amount = 0;
            for i in 0..num_threads {
                let (start, end) = intervals[i];
                res_ptrs.push(res_ptr);
                res_ptr.add((end - start) * res_inner_matrix_size);
                let mut prg: Vec<i64> = vec![0;
                iterate_shape.len()];
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
                a_ptr = new_a.data;
                b_ptr = new_b.data;
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
                let barrier = Arc::new(Barrier::new(num_threads + 1));
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
                    let barrier_clone = Arc::clone(&barrier);
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
                                    gemm::Parallelism::Rayon(threads)
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
                        barrier_clone.wait();
                    });
                }
                barrier.wait();
            });
            Ok(res)
        }
    }
}

#[allow(unused)]
pub(crate) fn matmul_no_out_concret<A, B, C>(
    lhs: &_Tensor<A>,
    rhs: &_Tensor<B>
)
    -> anyhow::Result<_Tensor<C>>
    where C: CommonBounds, A: CommonBounds + IntoScalar<C>, B: CommonBounds + IntoScalar<C>
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        if lhs.shape()[1] != rhs.shape()[0] {
            anyhow::bail!(
                "shape mismatch when trying to perform matrix multiplication got {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            );
        } else {
            let res: _Tensor<C>;
            res = _Tensor::zeros(vec![lhs.shape()[0], rhs.shape()[1]])?;
            let new_a: &_Tensor<C> = &lhs.try_astype()?;
            let new_b: &_Tensor<C> = &rhs.try_astype()?;
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
                    C::ZERO,
                    C::ONE,
                    false,
                    false,
                    false,
                    gemm::Parallelism::Rayon(rayon::current_num_threads())
                );
            }
            Ok(res)
        }
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
        if a_shape[a_shape.len() - 1] != b_shape[b_shape.len() - 2] {
            anyhow::bail!(
                "shape mismatch when trying to perform matrix multiplication got {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            );
        } else {
            let mut res_shape = predict_broadcast_shape(
                &a_shape[..a_shape.len() - 2],
                &b_shape[..b_shape.len() - 2],
                Location::caller(),
            )?.to_vec();
            let mut iterate_shape: Vec<i64> = res_shape.clone();
            res_shape.push(a_shape[a_shape.len() - 2]);
            res_shape.push(b_shape[b_shape.len() - 1]);
            let new_a: &_Tensor<C> = &lhs.try_astype::<C>()?;
            let new_b: &_Tensor<C> = &rhs.try_astype::<C>()?;
            let res: _Tensor<C>;
            res = _Tensor::zeros(res_shape)?;
            let a_strides: Vec<i64> = preprocess_strides(&a_shape, &lhs.strides());
            let b_strides: Vec<i64> = preprocess_strides(&b_shape, &rhs.strides());
            let len: usize = iterate_shape.iter().fold(1, |acc, x| acc * (*x as usize));
            let res_inner_matrix_size: usize =
                (res.shape()[res.shape().len() - 2] as usize) *
                (res.shape()[res.shape().len() - 1] as usize);
            iterate_shape.iter_mut().for_each(|x| {
                *x -= 1;
            });
            let mut a_ptr = new_a.data;
            let mut b_ptr = new_b.data;
            let mut res_ptr = res.data;
            let num_threads: usize = if len < rayon::current_num_threads() {
                len
            } else {
                rayon::current_num_threads()
            };
            let mut num_threads_each: Vec<usize> = if len < rayon::current_num_threads() {
                let vec: Vec<(usize, usize)> = mt_intervals(rayon::current_num_threads(), len);
                vec.iter()
                    .map(|x| x.1 - x.0)
                    .collect::<Vec<usize>>()
            } else {
                vec![1; rayon::current_num_threads()]
            };
            let intervals = mt_intervals(len, num_threads);
            let mut res_ptrs = Vec::with_capacity(num_threads);
            let mut a_ptrs = Vec::with_capacity(num_threads);
            let mut b_ptrs = Vec::with_capacity(num_threads);
            let mut prgs: Vec<Vec<i64>> = Vec::with_capacity(num_threads);
            let mut amount: usize = 0;
            for i in 0..num_threads {
                let (start, end) = intervals[i];
                res_ptrs.push(res_ptr);
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
                a_ptr = new_a.data;
                b_ptr = new_b.data;
                prgs.push(prg);
            }
            let lhs_cs: i64 = lhs.strides()[lhs.strides().len() - 1];
            let lhs_rs: i64 = lhs.strides()[lhs.strides().len() - 2];
            let dst_cs: i64 = res.strides()[res.strides().len() - 1];
            let dst_rs: i64 = res.strides()[res.strides().len() - 2];
            let rhs_cs: i64 = rhs.strides()[rhs.strides().len() - 1];
            let rhs_rs: i64 = rhs.strides()[rhs.strides().len() - 2];
            let m: usize = a_shape[a_shape.len() - 2] as usize;
            let n: usize = b_shape[b_shape.len() - 1] as usize;
            let k: usize = b_shape[b_shape.len() - 2] as usize;
            THREAD_POOL.with_borrow_mut(|pool: &mut threadpool::ThreadPool| {
                let barrier: Arc<Barrier> = Arc::new(Barrier::new(num_threads + 1));
                for i in (0..num_threads).rev() {
                    let threads: usize = num_threads_each.pop().unwrap();
                    let current_size: usize = intervals[i].1 - intervals[i].0;
                    let mut res_ptr = res_ptrs.pop().unwrap();
                    let mut a_ptr = a_ptrs.pop().unwrap();
                    let mut b_ptr = b_ptrs.pop().unwrap();
                    let mut prg: Vec<i64> = prgs.pop().unwrap();
                    let shape: Vec<i64> = iterate_shape.clone();
                    let __a_strides: Vec<i64> = a_strides.clone();
                    let __b_strides: Vec<i64> = b_strides.clone();
                    let barrier_clone: Arc<Barrier> = Arc::clone(&barrier);
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
                                    C::ZERO,
                                    C::ONE,
                                    false,
                                    false,
                                    false,
                                    gemm::Parallelism::Rayon(threads)
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
                        barrier_clone.wait();
                    });
                }
                barrier.wait();
            });
            Ok(res)
        }
    }
}

#[allow(unused)]
pub(crate) fn matmul_out_concret<A, B, C>(
    lhs: &_Tensor<A>,
    rhs: &_Tensor<B>,
    out: &_Tensor<C>
)
    -> anyhow::Result<_Tensor<C>>
    where C: CommonBounds, A: CommonBounds + IntoScalar<C>, B: CommonBounds + IntoScalar<C>
{
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        if lhs.shape()[1] != rhs.shape()[0] {
            anyhow::bail!(
                "shape mismatch when trying to perform matrix multiplication got {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            );
        } else {
            let res: _Tensor<C>;
            res = out.clone();
            let new_a: &_Tensor<C> = &lhs.try_astype()?;
            let new_b: &_Tensor<C> = &rhs.try_astype()?;
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
                    C::ZERO,
                    C::ONE,
                    false,
                    false,
                    false,
                    gemm::Parallelism::Rayon(rayon::current_num_threads())
                );
            }
            Ok(res)
        }
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
        if a_shape[a_shape.len() - 1] != b_shape[b_shape.len() - 2] {
            anyhow::bail!("shape mismatch");
        } else {
            let mut res_shape = predict_broadcast_shape(
                &a_shape[..a_shape.len() - 2],
                &b_shape[..b_shape.len() - 2],
                Location::caller(),
            )?.to_vec();
            let mut iterate_shape: Vec<i64> = res_shape.clone();
            res_shape.push(a_shape[a_shape.len() - 2]);
            res_shape.push(b_shape[b_shape.len() - 1]);
            let new_a: &_Tensor<C> = &lhs.try_astype::<C>()?;
            let new_b: &_Tensor<C> = &rhs.try_astype::<C>()?;
            let res: _Tensor<C>;
            res = out.clone();
            let a_strides: Vec<i64> = preprocess_strides(&a_shape, &lhs.strides());
            let b_strides: Vec<i64> = preprocess_strides(&b_shape, &rhs.strides());
            let len: usize = iterate_shape.iter().fold(1, |acc, x| acc * (*x as usize));
            let res_inner_matrix_size: usize =
                (res.shape()[res.shape().len() - 2] as usize) *
                (res.shape()[res.shape().len() - 1] as usize);
            iterate_shape.iter_mut().for_each(|x| {
                *x -= 1;
            });
            let mut a_ptr = new_a.data;
            let mut b_ptr = new_b.data;
            let mut res_ptr = res.data;
            let num_threads: usize = if len < rayon::current_num_threads() {
                len
            } else {
                rayon::current_num_threads()
            };
            let mut num_threads_each: Vec<usize> = if len < rayon::current_num_threads() {
                let vec: Vec<(usize, usize)> = mt_intervals(rayon::current_num_threads(), len);
                vec.iter()
                    .map(|x| x.1 - x.0)
                    .collect::<Vec<usize>>()
            } else {
                vec![1; rayon::current_num_threads()]
            };
            let intervals = mt_intervals(len, num_threads);
            let mut res_ptrs = Vec::with_capacity(num_threads);
            let mut a_ptrs = Vec::with_capacity(num_threads);
            let mut b_ptrs = Vec::with_capacity(num_threads);
            let mut prgs: Vec<Vec<i64>> = Vec::with_capacity(num_threads);
            let mut amount: usize = 0;
            for i in 0..num_threads {
                let (start, end) = intervals[i];
                res_ptrs.push(res_ptr);
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
                a_ptr = new_a.data;
                b_ptr = new_b.data;
                prgs.push(prg);
            }
            let lhs_cs: i64 = lhs.strides()[lhs.strides().len() - 1];
            let lhs_rs: i64 = lhs.strides()[lhs.strides().len() - 2];
            let dst_cs: i64 = res.strides()[res.strides().len() - 1];
            let dst_rs: i64 = res.strides()[res.strides().len() - 2];
            let rhs_cs: i64 = rhs.strides()[rhs.strides().len() - 1];
            let rhs_rs: i64 = rhs.strides()[rhs.strides().len() - 2];
            let m: usize = a_shape[a_shape.len() - 2] as usize;
            let n: usize = b_shape[b_shape.len() - 1] as usize;
            let k: usize = b_shape[b_shape.len() - 2] as usize;
            THREAD_POOL.with_borrow_mut(|pool: &mut threadpool::ThreadPool| {
                let barrier: Arc<Barrier> = Arc::new(Barrier::new(num_threads + 1));
                for i in (0..num_threads).rev() {
                    let threads: usize = num_threads_each.pop().unwrap();
                    let current_size: usize = intervals[i].1 - intervals[i].0;
                    let mut res_ptr = res_ptrs.pop().unwrap();
                    let mut a_ptr = a_ptrs.pop().unwrap();
                    let mut b_ptr = b_ptrs.pop().unwrap();
                    let mut prg: Vec<i64> = prgs.pop().unwrap();
                    let shape: Vec<i64> = iterate_shape.clone();
                    let __a_strides: Vec<i64> = a_strides.clone();
                    let __b_strides: Vec<i64> = b_strides.clone();
                    let barrier_clone: Arc<Barrier> = Arc::clone(&barrier);
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
                                    C::ZERO,
                                    C::ONE,
                                    false,
                                    false,
                                    false,
                                    gemm::Parallelism::Rayon(threads)
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
                        barrier_clone.wait();
                    });
                }
                barrier.wait();
            });
            Ok(res)
        }
    }
}
