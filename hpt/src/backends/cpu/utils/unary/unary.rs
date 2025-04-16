use crate::tensor_base::_Tensor;
use crate::{Tensor, THREAD_POOL};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape_utils::mt_intervals;
use hpt_iterator::iterator_traits::ParStridedIteratorSimdZip;
use hpt_iterator::TensorIterator;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::unary::Contiguous;
use hpt_traits::tensor::{CommonBounds, TensorInfo, TensorLike};
use hpt_types::dtype::TypeCommon;
use hpt_types::type_promote::{Eval, NormalOut};
use hpt_types::vectors::traits::*;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::borrow::Borrow;
use threadpool::ThreadPool;

pub fn unary_map<A, K, F, F2>(slice_a: &[A], slice_o: &mut [K], f: F, f2: F2)
where
    A: CommonBounds,
    K: CommonBounds,
    F: Fn(A::Vec) -> K::Vec + Sync + Send,
    F2: Fn(A) -> K + Sync + Send,
{
    if K::BYTE_SIZE == A::BYTE_SIZE {
        let mut chunk_o = slice_o.par_chunks_exact_mut(A::Vec::SIZE);
        let chunk_a = slice_a.par_chunks_exact(A::Vec::SIZE);
        chunk_o
            .remainder()
            .into_par_iter()
            .zip(chunk_a.remainder().into_par_iter())
            .for_each(|(out, buffer)| {
                *out = f2(*buffer);
            });
        chunk_o
            .into_par_iter()
            .zip(chunk_a.into_par_iter())
            .for_each(|(out, buffer)| {
                let out_ptr = out.as_mut_ptr() as *mut K::Vec;
                let buffer_ptr = buffer.as_ptr() as *const A::Vec;
                unsafe {
                    out_ptr.write_unaligned(f(buffer_ptr.read_unaligned()));
                }
            });
    } else {
        slice_o
            .par_iter_mut()
            .zip(slice_a.par_iter())
            .for_each(|(out, buffer)| {
                *out = f2(*buffer);
            });
    }
}

/// Perform unary operation with output tensor
pub fn unary_fn_with_out<A, O, K, F, F2, const DEVICE: usize, A2>(
    inp: &_Tensor<A, Cpu, DEVICE, A2>,
    f: F,
    f2: F2,
    out: Option<O>,
) -> std::result::Result<_Tensor<K, Cpu, DEVICE, A2>, TensorError>
where
    A: CommonBounds,
    K: CommonBounds,
    O: Borrow<_Tensor<K, Cpu, DEVICE, A2>>,
    F: Fn(A::Vec) -> K::Vec + Sync + Send,
    F2: Fn(A) -> K + Sync + Send,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    let mut ret = if let Some(out) = out {
        ShapeError::check_inplace_out_layout_valid(inp.shape(), &out.borrow().layout())?;
        out.borrow().static_cast()?
    } else {
        _Tensor::<K, Cpu, DEVICE, A2>::empty(inp.shape())?
    };
    if inp.parent().is_some() {
        ret.par_iter_mut_simd()
            .zip(inp.par_iter_simd())
            .for_each(|(a, b)| {
                *a = f2(b);
            });
        return Ok(ret);
    }
    unary_map(inp.as_raw(), ret.as_raw_mut(), f, f2);
    Ok(ret)
}

impl<T, A2, const DEVICE: usize> _Tensor<T, Cpu, DEVICE, A2>
where
    T: CommonBounds + Eval,
    <T as Eval>::Output: CommonBounds,
    T::Vec: Eval<Output = <<T as Eval>::Output as TypeCommon>::Vec>,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    pub fn is_inf(
        &self,
    ) -> std::result::Result<_Tensor<<T as Eval>::Output, Cpu, DEVICE, A2>, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._is_inf(),
            |x| x._is_inf(),
            None::<_Tensor<<T as Eval>::Output, Cpu, DEVICE, A2>>,
        )
    }

    pub fn is_nan(
        &self,
    ) -> std::result::Result<_Tensor<<T as Eval>::Output, Cpu, DEVICE, A2>, TensorError> {
        unary_fn_with_out(
            self,
            |x| x._is_nan(),
            |x| x._is_nan(),
            None::<_Tensor<<T as Eval>::Output, Cpu, DEVICE, A2>>,
        )
    }
}

pub(crate) fn cumulate<
    T: CommonBounds,
    F: Fn(T, T) -> T + Send + Sync + 'static + Copy,
    A: Into<Option<i64>>,
    const DEVICE: usize,
    A2,
>(
    a: &_Tensor<T, Cpu, DEVICE, A2>,
    axis: A,
    init_val: T,
    op: F,
) -> std::result::Result<_Tensor<T, Cpu, DEVICE, A2>, TensorError>
where
    T: NormalOut<T, Output = T>,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    match axis.into() {
        Some(axis) => {
            let mut _axis = axis;
            if _axis < 0 {
                _axis += a.ndim() as i64;
            }
            ShapeError::check_index_out_of_range(_axis, a.ndim() as i64)?;
            let stride = a.strides()[_axis as usize];
            let inner_loop = a.shape()[_axis as usize] as usize;
            let outer_loop = a.size() / inner_loop;
            let mut shape = a.shape().to_vec();
            shape.iter_mut().for_each(|x| {
                *x -= 1;
            });
            shape.swap(_axis as usize, a.shape().len() - 1);
            let mut strides = a.strides().to_vec();
            strides.swap(_axis as usize, a.strides().len() - 1);
            let res = a.empty_like()?;
            let res_stride = res.strides()[_axis as usize];
            let mut res_strides = res.strides().to_vec();
            res_strides.swap(_axis as usize, res.strides().len() - 1);
            THREAD_POOL.with_borrow_mut(|pool: &mut ThreadPool| {
                let num_threads;
                if outer_loop < pool.max_count() {
                    num_threads = outer_loop;
                } else {
                    num_threads = pool.max_count();
                }
                let mut intervals = mt_intervals(outer_loop, num_threads);
                let mut prgs = Vec::with_capacity(num_threads);
                let mut ptrs = Vec::with_capacity(num_threads);
                let mut res_ptrs = Vec::with_capacity(num_threads);
                let mut shapes = Vec::with_capacity(num_threads);
                let mut __res_strides = Vec::with_capacity(num_threads);
                let mut __inp_strides = Vec::with_capacity(num_threads);
                for i in 0..num_threads {
                    let (start, _) = intervals[i];
                    let mut prg_tmp = vec![0; a.shape().len()];
                    let mut ptr_tmp = a.ptr();
                    let mut res_ptr_tmp = res.ptr();
                    let mut amount = (start as i64) * (shape[shape.len() - 1] + 1);
                    let mut inp_amount = 0i64;
                    let mut res_amount = 0i64;
                    for j in (0..a.shape().len() as i64).rev() {
                        prg_tmp[j as usize] = amount % (shape[j as usize] + 1);
                        amount /= shape[j as usize] + 1;
                        inp_amount += prg_tmp[j as usize] * strides[j as usize];
                        res_amount += prg_tmp[j as usize] * res_strides[j as usize];
                    }
                    res_ptr_tmp.offset(res_amount);
                    ptr_tmp.offset(inp_amount);
                    prgs.push(prg_tmp);
                    ptrs.push(ptr_tmp);
                    res_ptrs.push(res_ptr_tmp);
                    shapes.push(shape.clone());
                    __res_strides.push(res_strides.clone());
                    __inp_strides.push(strides.clone());
                }
                for _ in 0..num_threads {
                    let (start, end) = intervals.pop().unwrap();
                    let mut prg = prgs.pop().unwrap();
                    let mut ptr = ptrs.pop().unwrap();
                    let mut res_ptr = res_ptrs.pop().unwrap();
                    let current_size = end - start;
                    let __shape = shapes.pop().unwrap();
                    let __res_strides = __res_strides.pop().unwrap();
                    let __strides = __inp_strides.pop().unwrap();
                    pool.execute(move || {
                        for _ in 0..current_size {
                            let mut tmp = init_val;
                            for i in 0..inner_loop as i64 {
                                tmp = op(tmp, ptr[i * stride]);
                                res_ptr[i * res_stride] = tmp;
                            }
                            for j in (0..(__shape.len() as i64) - 1).rev() {
                                let j = j as usize;
                                if prg[j] < __shape[j] {
                                    prg[j] += 1;
                                    res_ptr.offset(__res_strides[j]);
                                    ptr.offset(__strides[j]);
                                    break;
                                } else {
                                    prg[j] = 0;
                                    res_ptr.offset(-__shape[j] * __res_strides[j]);
                                    ptr.offset(-__shape[j] * __strides[j]);
                                }
                            }
                        }
                    });
                }
                pool.join();
            });
            Ok(res)
        }
        None => {
            let mut res = _Tensor::<T, Cpu, DEVICE, A2>::empty(vec![a.size() as i64])?;
            let mut tmp = init_val;
            if a.is_contiguous() {
                let raw = a.as_raw();
                let res_raw = res.as_raw_mut();
                for i in 0..a.size() {
                    tmp = op(tmp, raw[i]);
                    res_raw[i] = tmp;
                }
                Ok(res)
            } else {
                let new_self = a.contiguous()?;
                let raw = new_self.as_raw();
                let mut tmp = init_val;
                let res_raw = res.as_raw_mut();
                for i in 0..a.size() {
                    tmp = op(tmp, raw[i]);
                    res_raw[i] = tmp;
                }
                Ok(res)
            }
        }
    }
}

impl<T, A2, const DEVICE: usize> Tensor<T, Cpu, DEVICE, A2>
where
    T: CommonBounds + Eval,
    <T as Eval>::Output: CommonBounds,
    T::Vec: Eval<Output = <<T as Eval>::Output as TypeCommon>::Vec>,
    A2: Allocator,
    A2::Output: AllocatorOutputRetrive,
{
    /// Checks for infinity (`inf`) values in the tensor.
    ///
    /// This method returns a new tensor where each element indicates whether the corresponding element
    /// in the input tensor is an infinity value (`+inf` or `-inf`). The output tensor will contain boolean-like values
    /// (1 for `inf`, 0 for non-`inf`).
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor of type `_Tensor<<T as Eval>::Output>`,
    /// where each element is either `1` (if the corresponding element is `inf`) or `0` (if it is not).
    pub fn is_inf(&self) -> Result<Tensor<<T as Eval>::Output, Cpu, DEVICE, A2>, TensorError> {
        Ok(self.inner.is_inf()?.into())
    }

    /// Checks for `NaN` (Not-a-Number) values in the tensor.
    ///
    /// This method returns a new tensor where each element indicates whether the corresponding element
    /// in the input tensor is a `NaN` value. The output tensor will contain boolean-like values
    /// (1 for `NaN`, 0 for non-`NaN`).
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor of type `_Tensor<<T as Eval>::Output>`,
    /// where each element is either `1` (if the corresponding element is `NaN`) or `0` (if it is not).
    pub fn is_nan(&self) -> Result<Tensor<<T as Eval>::Output, Cpu, DEVICE, A2>, TensorError> {
        Ok(self.inner.is_nan()?.into())
    }
}
