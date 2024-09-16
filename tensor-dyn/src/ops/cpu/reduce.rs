use crate::backend::Cpu;
use crate::ops::cpu::kernels::reduce_kernels::fast_reduce_no_simd;
use crate::ops::cpu::reduce_template::reduce_template;
use crate::slice::SliceOps;
use crate::tensor_base::_Tensor;
use crate::{argmax_kernel, argmin_kernel};

use crate::THREAD_POOL;
use anyhow;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
use std::borrow::BorrowMut;
use std::sync::Arc;
use std::sync::Barrier;
use tensor_common::axis::{process_axes, Axis};
use tensor_common::pointer::Pointer;
use tensor_common::shape::Shape;
use tensor_common::shape_utils::{mt_intervals, mt_intervals_simd};
use tensor_common::slice::Slice;
use tensor_common::strides::Strides;
use tensor_iterator::iterator_traits::StridedIterator;
use tensor_traits::shape_manipulate::ShapeManipulate;
use tensor_traits::tensor::CommonBounds;
use tensor_traits::tensor::TensorCreator;
use tensor_traits::tensor::{IndexReduce, TensorInfo};
use tensor_types::convertion::Convertor;
use tensor_types::dtype::TypeCommon;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::{Cmp, NormalOut};

#[derive(Debug, Clone)]
pub(crate) struct ReductionPreprocessor<T, U> {
    pub ptrs: Pointer<T>,
    pub res_ptrs: Pointer<U>,
    pub strides: Strides,
    pub start: usize,
    pub end: usize,
    pub prg: Vec<i64>,
    pub a_prg: Vec<i64>,
    pub shape: Shape,
    pub a_shape: Shape,
}

impl<T, U> ReductionPreprocessor<T, U>
where
    T: Clone,
    U: Clone,
{
    pub fn new(
        num_threads: usize,
        loop_size: usize,
        inner_loop_size: usize,
        ptrs: Pointer<T>,
        mut res_ptrs: Pointer<U>,
        strides: Strides,
        a_shape: Shape,
        transposed_shape: Shape,
        res_shape: Shape,
    ) -> Vec<ReductionPreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators: Vec<ReductionPreprocessor<T, U>> = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; res_shape.len()];
        let res_ptrs = res_ptrs.borrow_mut();
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();

            /*traverse the whole result shape and increment the input data ptr based on current thread id*/
            for i in (0..=res_shape.len() - 1).rev() {
                a_data_ptr_cpy.offset(progress_init_a_data[i] * strides[i]);
            }
            // calculate the total task amount so far based on current thread id,
            // we are splitting the whole tensor into two axes
            // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]               thread 0
            // [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]     thread 1
            // [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]     thread 2
            // [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]     thread 3
            // [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]     thread 4
            // where the first axis is where we are splitting the tensor
            let mut tmp1 = (task_amout * inner_loop_size) as i64;
            let mut prg = vec![0; a_shape.len() - 1]; /* -1 because we want to escape the last axis */

            // since the axis we want to reduce include the most inner axis, we will skip the iteration of the last axis
            // so we use (0..=a_shape.len() - 2).rev()
            for i in (0..=a_shape.len() - 2).rev() {
                prg[i] = tmp1 % transposed_shape[i];
                tmp1 /= transposed_shape[i];
            }

            // increment the res ptr based on the current thread task amount for next thread (next iteration)
            task_amout += intervals[id].1 - intervals[id].0;
            let res_ptr_cpy = res_ptrs.clone();
            res_ptrs.add(intervals[id].1 - intervals[id].0);

            let mut tmp2 = task_amout as i64;
            for j in (0..=res_shape.len() - 1).rev() {
                progress_init_a_data[j] = tmp2 % res_shape[j];
                tmp2 /= res_shape[j];
            }
            iterators.push(ReductionPreprocessor {
                ptrs: a_data_ptr_cpy.clone(),
                res_ptrs: res_ptr_cpy,
                strides: strides.clone(),
                start: intervals[id].0,
                end: intervals[id].1,
                prg,
                a_prg: vec![],
                shape: res_shape.clone(),
                a_shape: a_shape.clone(),
            });
        }
        iterators
    }

    pub fn new2(
        num_threads: usize,
        loop_size: usize,
        inner_loop_size: usize,
        ptrs: Pointer<T>,
        mut res_ptrs: Pointer<U>,
        transposed_strides: Strides,
        transposed_shape: Shape,
        res_shape: Shape,
    ) -> Vec<ReductionPreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; res_shape.len()];
        let res_ptrs = res_ptrs.borrow_mut();
        let ndim = res_shape.len() as i64;

        // [0, 6, 12, 18, 24, 30] res0    thread 0
        // [1, 7, 13, 19, 25, 31] res1    thread 1
        // [2, 8, 14, 20, 26, 32] res0    thread 0
        // [3, 9, 15, 21, 27, 33] res1    thread 1
        // [4, 10, 16, 22, 28, 34] res0   thread 0
        // [5, 11, 17, 23, 29, 35] res1   thread 1
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();

            for i in (0..ndim - 1).rev() {
                a_data_ptr_cpy
                    .offset(progress_init_a_data[i as usize] * transposed_strides[i as usize]);
            }

            let progress_init_a_data_cpy = progress_init_a_data.clone();

            task_amout += intervals[id].1 - intervals[id].0;

            let prg = vec![0; transposed_shape.len()];

            let res_ptr_cpy = res_ptrs.clone();
            res_ptrs.add((intervals[id].1 - intervals[id].0) * inner_loop_size);

            let mut tmp = task_amout as i64;
            for j in (0..ndim - 1).rev() {
                progress_init_a_data[j as usize] = tmp % res_shape[j as usize];
                tmp /= res_shape[j as usize];
            }

            iterators.push(ReductionPreprocessor {
                ptrs: a_data_ptr_cpy.clone(),
                res_ptrs: res_ptr_cpy,
                strides: transposed_strides.clone(),
                start: intervals[id].0,
                end: intervals[id].1,
                prg,
                a_prg: progress_init_a_data_cpy,
                shape: res_shape.clone(),
                a_shape: transposed_shape.clone(),
            });
        }
        iterators
    }
}

macro_rules! init_arr {
    (
        $result:ident,
        $shape:ident,
        $macro_init_val:expr,
        $($specific_type:tt)*
    ) => {
        $result = _Tensor::<$($specific_type)*, Cpu>::empty($shape.clone())?;
        $result.as_raw_mut().par_iter_mut().for_each(|x| {
            *x = $macro_init_val;
        });
    };
}

macro_rules! body_one_axis {
    (
        $axes:ident,
        $a:ident,
        $init_val:ident,
        $keepdims:ident,
        $c:ident,
        $kernel_name:ident,
        $generic_a:ident,
        $($specific_type:tt)*
    ) => {
        let a_: &_Tensor<$generic_a> = &$a;
        let a_shape = a_.shape();
        let a_shape_tmp = a_shape.clone();
        let mut a_shape_cpy = a_shape_tmp.to_vec();
        for axis in $axes.iter() {
            a_shape_cpy[*axis] = 0;
        }
        let mut j = a_.ndim() - $axes.len();
        let mut k = 0;
        let mut track_idx = 0;
        let mut transposed_axis = vec![0; a_.ndim()];
        for i in 0..a_.ndim() {
            if a_shape_cpy[i] != 0 {
                transposed_axis[k] = i;
                k += 1;
            } else {
                transposed_axis[j] = $axes[track_idx];
                j += 1;
                track_idx += 1;
            }
        }
        let transposed_tensor = a_.permute(transposed_axis)?;
        let transposed_strides = transposed_tensor.strides().clone();
        let transposed_shape = transposed_tensor.shape().clone();
        let mut transposed_shape_cpy = transposed_shape.clone();
        transposed_shape_cpy.iter_mut().for_each(|x| {
            *x -= 1;
        });
        let a_data = a_.ptr();
        let mut res_shape = Vec::with_capacity(a_.ndim() - $axes.len());
        a_shape_cpy.iter().for_each(|x| {
            (if *x != 0 {
                res_shape.push(*x)
            })
        });
        let mut new_shape: Option<Vec<i64>> = None;
        let result;
        let result_size;
        if $keepdims {
            let mut shape_tmp = Vec::with_capacity(a_.ndim());
            a_shape_cpy.iter().for_each(|x| {
                (if *x != 0 {
                    shape_tmp.push(*x);
                } else {
                    shape_tmp.push(1);
                })
            });
            new_shape = Some(shape_tmp);
        }
        let res_shape = Arc::new(res_shape);
        if let Some(out) = $c {
            if let Some(s) = &new_shape {
                for i in 0..a_.ndim() {
                    if s[i] != out.shape()[i] {
                        return Err(anyhow::Error::msg(format!(
                            "Output array has incorrect shape"
                        )));
                    }
                }
            } else {
                for i in 0..a_.ndim() - $axes.len() {
                    if a_shape_cpy[i] != out.shape()[i] {
                        return Err(anyhow::Error::msg(format!(
                            "Output array has incorrect shape"
                        )));
                    }
                }
            }
            result = out;
            result_size = result.size();
        } else {
            init_arr!(result, res_shape, $init_val, $($specific_type)*);
            result_size = result.size();
        }
        let result_data = result.ptr();
        let a_last_index = a_.ndim() - 1;
        let inner_loop_size = transposed_shape[a_last_index];
        let a_data_ptr = a_data.clone();
        let last_stride = transposed_strides[a_last_index];
        THREAD_POOL.with_borrow_mut(|pool| {
            let num_threads;
            if result_size < pool.max_count() {
                num_threads = result_size;
            } else {
                num_threads = pool.max_count();
            }
            let mut iterators = ReductionPreprocessor::new(
                num_threads,
                result_size,
                1,
                a_data_ptr,
                result_data,
                transposed_strides,
                transposed_shape_cpy.into(),
                transposed_shape,
                res_shape.into(),
            );
            let barrier = Arc::new(Barrier::new(num_threads + 1));
            for _ in (0..num_threads).rev() {
                let mut iterator = iterators.pop().unwrap();
                let mut result_ptr_c = iterator.res_ptrs;
                let mut a_data_ptr = iterator.ptrs;
                let current_size = iterator.end - iterator.start;
                let barrier_clone = Arc::clone(&barrier);
                pool.execute(move || {
                    let shape_len = iterator.a_shape.len() as i64;
                    for _ in 0..current_size {
                        $kernel_name!(
                            init_val,
                            iterator,
                            inner_loop_size,
                            inner_loop_size,
                            result_ptr_c,
                            a_data_ptr,
                            last_stride,
                            shape_len
                        );
                    }
                    barrier_clone.wait();
                });
            }
            barrier.wait();
        });
        if let Some(new_shape) = new_shape {
            let result = result.reshape(new_shape)?;
            return Ok(result);
        } else {
            return Ok(result);
        }
    };
}

macro_rules! register_reduction_one_axis {
    (
        $generic_a:ident,
        $generic_b:ident,
        $fn_name:ident,
        $kernel_name:ident,
        $($trait_bound:tt)*
    ) => {
        #[cfg_attr(feature = "track_caller", track_caller)]
        pub(crate) fn $fn_name<$generic_a, $generic_b>(a: &_Tensor<$generic_a>, axes: Vec<usize>,
             init_val: $generic_b, keepdims: bool, c: Option<_Tensor<$generic_b>>) -> anyhow::Result<_Tensor<$generic_b>> $($trait_bound)*
         {
            body_one_axis!(axes, a, init_val, keepdims, c, $kernel_name, $generic_a, $generic_b);
        }
    };
    (
        $generic_a:ident,
        $fn_name:ident,
        $kernel_name:ident,
        $($trait_bound:tt)*
    ) => {
        #[cfg_attr(feature = "track_caller", track_caller)]
        pub(crate) fn $fn_name<$generic_a>(a: &_Tensor<$generic_a>, axes: Vec<usize>,
             init_val: $generic_a, keepdims: bool, c: Option<_Tensor<$generic_a>>) -> anyhow::Result<_Tensor<$generic_a>> $($trait_bound)*
         {
            body_one_axis!(axes, a, init_val, keepdims, c, $kernel_name, $generic_a, $generic_a);
        }
    };
    (
        $generic_a:ident => [$($specific_type:tt)*],
        $fn_name:ident,
        $kernel_name:ident,
        $($trait_bound:tt)*
    ) => {
        #[cfg_attr(feature = "track_caller", track_caller)]
        pub(crate) fn $fn_name<$generic_a>(a: &_Tensor<$generic_a>, axes: Vec<usize>,
             init_val: $($specific_type)*, keepdims: bool, c: Option<_Tensor<$($specific_type)*>>) -> anyhow::Result<_Tensor<$($specific_type)*>> $($trait_bound)*
         {
            body_one_axis!(axes, a, init_val, keepdims, c, $kernel_name, $generic_a, $($specific_type)*);
        }
    };
}

use tensor_types::vectors::traits::*;

use super::reduce_template::uncontiguos_reduce_template;
use super::uncontiguous_reduce;

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn reduce<T, F, F2>(
    a: &_Tensor<T>,
    op: F,
    vec_op: F2,
    axes: &[usize],
    init_val: T,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<T>>,
) -> anyhow::Result<_Tensor<T>>
where
    T: CommonBounds + IntoScalar<T> + Convertor,
    F: Fn(T, T) -> T + Sync + Send + 'static + Copy,
    F2: Fn(<T as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec
        + Sync
        + Send
        + 'static
        + Copy,
    <T as TypeCommon>::Vec: Copy,
{
    if a.is_contiguous() {
        contiguous_reduce::<
            _,
            _,
            _,
            fn(T) -> T,
            _,
            fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec,
            T,
        >(
            a, op, op, None, vec_op, None, &axes, init_val, keepdims, init_out, c,
        )
    } else {
        uncontiguous_reduce::<
            _,
            _,
            _,
            fn(T) -> T,
            _,
            fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec,
            T,
        >(
            a, op, op, None, vec_op, None, &axes, init_val, keepdims, init_out, c,
        )
    }
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn reduce2<T, F, F2, F3, O>(
    a: &_Tensor<T>,
    op: F,
    op2: F2,
    vec_op: F3,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O>>,
) -> anyhow::Result<_Tensor<O>>
where
    T: CommonBounds + IntoScalar<O> + Convertor,
    F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec
        + Sync
        + Send
        + 'static
        + Copy,
    O: CommonBounds,
    <T as TypeCommon>::Vec: Copy,
    <O as TypeCommon>::Vec: Copy,
{
    if a.is_contiguous() {
        contiguous_reduce::<
            T,
            F,
            F2,
            fn(O) -> O,
            _,
            fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec,
            O,
        >(
            a, op, op2, None, vec_op, None, &axes, init_val, keepdims, init_out, c,
        )
    } else {
        uncontiguous_reduce::<
            T,
            F,
            F2,
            fn(O) -> O,
            _,
            fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec,
            O,
        >(
            a, op, op2, None, vec_op, None, &axes, init_val, keepdims, init_out, c,
        )
    }
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn reduce3<T, F, F2, F3, F4, F5, O>(
    a: &_Tensor<T>,
    op: F,
    op2: F2,
    op3: F3,
    op4: F4,
    op5: F5,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O>>,
) -> anyhow::Result<_Tensor<O>>
where
    T: CommonBounds + IntoScalar<O> + Convertor,
    F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec
        + Sync
        + Send
        + 'static
        + Copy,
    F5: Fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec + Sync + Send + 'static + Copy,
    O: CommonBounds,
    <O as TypeCommon>::Vec: Copy,
{
    if a.is_contiguous() {
        contiguous_reduce::<T, F, F2, F3, F4, F5, O>(
            a,
            op,
            op2,
            Some(op3),
            op4,
            Some(op5),
            &axes,
            init_val,
            keepdims,
            init_out,
            c,
        )
    } else {
        uncontiguous_reduce::<T, F, F2, F3, F4, F5, O>(
            a,
            op,
            op2,
            Some(op3),
            op4,
            Some(op5),
            &axes,
            init_val,
            keepdims,
            init_out,
            c,
        )
    }
}

register_reduction_one_axis!(
    T => [i64],
    argmax,
    argmax_kernel,
    where T: CommonBounds + NormalOut<T, Output = T> + Cmp
);

register_reduction_one_axis!(
    T => [i64],
    argmin,
    argmin_kernel,
    where T: CommonBounds + NormalOut<T, Output = T> + Cmp
);

impl<T: CommonBounds + NormalOut<Output = T> + Cmp> IndexReduce for _Tensor<T> {
    type Output = _Tensor<i64>;

    fn argmax<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        argmax(self, axes, 0, keep_dims, None)
    }

    fn argmin<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        argmin(self, axes, 0, keep_dims, None)
    }
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn contiguous_reduce<T, F, F2, F3, F4, F5, O>(
    a: &_Tensor<T>,
    op: F,
    op2: F2,
    op3: Option<F3>,
    vec_op: F4,
    vec_post: Option<F5>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O>>,
) -> anyhow::Result<_Tensor<O>>
where
    T: CommonBounds + IntoScalar<O> + Convertor,
    O: CommonBounds,
    F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec
        + 'static
        + Copy
        + Send
        + std::marker::Sync,
    F5: Fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec + Sync + Send + 'static + Copy,
    <T as TypeCommon>::Vec: Copy,
    <O as TypeCommon>::Vec: Copy,
{
    reduce_template(
        a,
        axes,
        init_val,
        keepdims,
        init_out,
        c,
        move |res| {
            let val = a
                .as_raw_mut()
                .par_iter()
                .fold(|| init_val, |acc, &x| op(acc, x))
                .reduce(|| init_val, |a, b| op2(a, b));
            if let Some(op3) = op3 {
                *res = op3(op2(val, *res));
            } else {
                *res = op2(val, *res);
            }
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let iterators = ReductionPreprocessor::new(
                num_threads,
                result.size(),
                inner_loop_size_2,
                a.ptr(),
                result.ptr(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                result.shape().clone(),
            );
            iterators.into_par_iter().for_each(|mut iterator| {
                let mut result_ptr_c = iterator.res_ptrs.clone();
                let mut a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.a_shape.len() as i64;
                for _ in 0..current_size {
                    for _ in 0..inner_loop_size_2 {
                        let mut tmp = result_ptr_c[0isize];
                        for i in 0..inner_loop_size as i64 {
                            let a_val = a_data_ptr[i];
                            tmp = op(tmp, a_val);
                        }
                        result_ptr_c[0isize] = tmp;
                        for j in (0..shape_len - 1).rev() {
                            if iterator.prg[j as usize] < iterator.a_shape[j as usize] {
                                iterator.prg[j as usize] += 1;
                                a_data_ptr.offset(iterator.strides[j as usize]);
                                break;
                            } else {
                                iterator.prg[j as usize] = 0;
                                a_data_ptr.offset(
                                    -iterator.strides[j as usize] * iterator.a_shape[j as usize],
                                );
                            }
                        }
                    }
                    if let Some(op3) = op3 {
                        let tmp = result_ptr_c[0isize];
                        let tmp = op3(tmp);
                        result_ptr_c[0isize] = tmp;
                    }
                    result_ptr_c.add(1);
                }
            });
        },
        move |num_threads, inner_loop_size, result| {
            let intervals = mt_intervals_simd(inner_loop_size, num_threads, O::Vec::SIZE);
            let mut slices = vec![Slice::Full; a.ndim()];
            let mut slices_res = vec![Slice::Full; result.ndim()];
            let mut sliced_tensors = Vec::with_capacity(num_threads);
            let mut sliced_res = Vec::with_capacity(num_threads);
            assert_eq!(inner_loop_size, result.size());
            for (start, end) in intervals.into_iter() {
                if end - start == 0 {
                    continue;
                }
                slices[(a.ndim()) - 1] = Slice::Range((start as i64, end as i64));
                slices_res[(result.ndim()) - 1] = Slice::Range((start as i64, end as i64));
                sliced_tensors.push(a.slice(&slices).expect("Slice failed"));
                sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
            }
            sliced_tensors
                .into_par_iter()
                .zip(sliced_res.into_par_iter())
                .for_each(move |(inp, res)| {
                    let inp_ptr = inp.ptr();
                    let res_ptr = res.ptr();
                    #[cfg(feature = "simd")]
                    {
                        let inner_loop_size = *res.shape().last().unwrap() as isize;
                        let outer_loop_size = (inp.size() as isize) / inner_loop_size;
                        if *inp.strides().last().unwrap() == 1 && O::Vec::SIZE == T::Vec::SIZE {
                            use crate::ops::cpu::kernels::reduce_kernels::fast_reduce_simd;
                            fast_reduce_simd(
                                inner_loop_size,
                                outer_loop_size,
                                inp_ptr,
                                res_ptr,
                                inp.strides().inner(),
                                inp.shape().inner(),
                                O::Vec::SIZE as isize,
                                op,
                                vec_op,
                                op3,
                                vec_post,
                            );
                        } else {
                            fast_reduce_no_simd(
                                inner_loop_size,
                                outer_loop_size,
                                inp_ptr,
                                res_ptr,
                                inp.strides().inner(),
                                inp.shape().inner(),
                                op,
                                op3,
                            );
                        }
                    }
                    #[cfg(not(feature = "simd"))]
                    {
                        res.iter_mut().zip(inp.iter()).for_each(|(x, y)| {
                            *x = op(*x, y);
                        });
                        if let Some(op3) = op3 {
                            res.iter_mut().for_each(|x| {
                                *x = op3(*x);
                            });
                        }
                    }
                });
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let outer_loop_size = result.size() / inner_loop_size;
            let iterators = ReductionPreprocessor::new2(
                num_threads,
                outer_loop_size,
                inner_loop_size,
                a.ptr(),
                result.ptr(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                result.shape().clone(),
            );
            iterators.into_par_iter().for_each(|iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.shape.len() as i64;
                #[cfg(feature = "simd")]
                {
                    use crate::ops::cpu::kernels::reduce_kernels::reduce_dim_not_include_simd;
                    let inp_strides = &iterator.strides;
                    let inp_shape = &iterator.a_shape;
                    let mut prg1 = iterator.prg.clone();
                    let mut prg2 = iterator.a_prg.clone();
                    reduce_dim_not_include_simd(
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr,
                        result_ptr_c,
                        &inp_strides,
                        &inp_shape,
                        &mut prg1,
                        &mut prg2,
                        shape_len,
                        op,
                        op3,
                        vec_op,
                        vec_post,
                    );
                }

                #[cfg(not(feature = "simd"))]
                {
                    let mut result_ptr_c = result_ptr_c;
                    let mut a_data_ptr = a_data_ptr;
                    for _i in 0..current_size {
                        for _ in 0..inner_loop_size_2 {
                            for i in 0..inner_loop_size as i64 {
                                let a_val = a_data_ptr[i];
                                let result_val = result_ptr_c[i];
                                let mut_ref = unsafe { &mut *result_ptr_c.ptr.offset(i as isize) };
                                *mut_ref = op(result_val, a_val);
                            }
                            for j in (shape_len..=(iterator.a_shape.len() as i64) - 1).rev() {
                                if iterator.prg[j as usize] < iterator.a_shape[j as usize] {
                                    iterator.prg[j as usize] += 1;
                                    a_data_ptr.offset(iterator.strides[j as usize]);
                                    break;
                                } else {
                                    iterator.prg[j as usize] = 0;
                                    a_data_ptr.offset(
                                        -iterator.strides[j as usize]
                                            * iterator.a_shape[j as usize],
                                    );
                                }
                            }
                        }
                        for j in (0..shape_len - 1).rev() {
                            if iterator.a_prg[j as usize] < iterator.a_shape[j as usize] {
                                iterator.a_prg[j as usize] += 1;
                                a_data_ptr.offset(iterator.strides[j as usize]);
                                break;
                            } else {
                                iterator.a_prg[j as usize] = 0;
                                a_data_ptr.offset(
                                    -iterator.strides[j as usize] * iterator.a_shape[j as usize],
                                );
                            }
                        }
                        if let Some(op3) = op3 {
                            for i in 0..inner_loop_size as i64 {
                                let result_val = result_ptr_c[i];
                                let mut_ref = unsafe { &mut *result_ptr_c.ptr.offset(i as isize) };
                                *mut_ref = op3(result_val);
                            }
                        }
                        result_ptr_c.add(inner_loop_size);
                        iterator.prg.iter_mut().for_each(|x| {
                            *x = 0;
                        });
                    }
                }
            });
        },
    )
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn uncontiguous_reduce<T, F, F2, F3, F4, F5, O>(
    a: &_Tensor<T>,
    op: F,
    op2: F2,
    op3: Option<F3>,
    _: F4,
    _: Option<F5>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O>>,
) -> anyhow::Result<_Tensor<O>>
where
    T: CommonBounds + IntoScalar<O> + Convertor,
    O: CommonBounds,
    F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec
        + 'static
        + Copy
        + Send
        + std::marker::Sync,
    F5: Fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec + Sync + Send + 'static + Copy,
    <T as TypeCommon>::Vec: Copy,
    <O as TypeCommon>::Vec: Copy,
{
    uncontiguos_reduce_template(
        a,
        axes,
        init_val,
        keepdims,
        init_out,
        c,
        move |res| {
            let val = a
                .as_raw_mut()
                .par_iter()
                .fold(|| init_val, |acc, &x| op(acc, x))
                .reduce(|| init_val, |a, b| op2(a, b));
            if let Some(op3) = op3 {
                *res = op3(op2(val, *res));
            } else {
                *res = op2(val, *res);
            }
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let a_last_stride = transposed_tensor.strides()[a.ndim() - 1];
            let iterators = uncontiguous_reduce::ReductionPreprocessor::new(
                num_threads,
                result.size(),
                inner_loop_size_2,
                a.ptr(),
                result.ptr(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                result.shape().clone(),
                result.strides().inner(),
            );
            let res_shape = result.shape().clone();
            iterators.into_par_iter().for_each(|mut iterator| {
                let mut result_ptr_c = iterator.res_ptrs;
                let mut a_data_ptr = iterator.ptrs;
                let current_size = iterator.end - iterator.start;
                let res_shape = res_shape.clone();
                let res_strides = result.strides().clone();
                let shape_len = iterator.a_shape.len() as i64;
                for _ in 0..current_size {
                    for _ in 0..inner_loop_size_2 {
                        let mut tmp = result_ptr_c[0isize];
                        for i in 0..inner_loop_size as i64 {
                            let a_val = a_data_ptr[i * a_last_stride];
                            tmp = op(tmp, a_val);
                        }
                        result_ptr_c[0isize] = tmp;
                        for j in (0..shape_len - 1).rev() {
                            if iterator.prg[j as usize] < iterator.a_shape[j as usize] {
                                iterator.prg[j as usize] += 1;
                                a_data_ptr.offset(iterator.strides[j as usize]);
                                break;
                            } else {
                                iterator.prg[j as usize] = 0;
                                a_data_ptr.offset(
                                    -iterator.strides[j as usize] * iterator.a_shape[j as usize],
                                );
                            }
                        }
                    }
                    if let Some(op3) = op3 {
                        result_ptr_c[0isize] = op3(result_ptr_c[0isize]);
                    }
                    for j in (0..res_shape.len()).rev() {
                        if iterator.res_prg[j] < res_shape[j] - 1 {
                            iterator.res_prg[j] += 1;
                            result_ptr_c.offset(res_strides[j]);
                            break;
                        } else {
                            iterator.res_prg[j] = 0;
                            result_ptr_c.offset(-res_strides[j] * (res_shape[j] - 1));
                        }
                    }
                }
            });
        },
        move |num_threads, inner_loop_size, ap, result| {
            let intervals = mt_intervals_simd(inner_loop_size, num_threads, O::Vec::SIZE);
            let mut slices = vec![Slice::Full; ap.ndim()];
            let mut slices_res = vec![Slice::Full; result.ndim()];
            let mut sliced_tensors = Vec::with_capacity(num_threads);
            let mut sliced_res = Vec::with_capacity(num_threads);
            assert_eq!(inner_loop_size, result.size());
            for (start, end) in intervals.into_iter() {
                if end - start == 0 {
                    continue;
                }
                slices[(ap.ndim()) - 1] = Slice::Range((start as i64, end as i64));
                slices_res[(result.ndim()) - 1] = Slice::Range((start as i64, end as i64));
                sliced_tensors.push(ap.slice(&slices).expect("Slice failed"));
                sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
            }
            sliced_tensors
                .into_par_iter()
                .zip(sliced_res.into_par_iter())
                .for_each(move |(inp, res)| {
                    res.iter_mut().zip(inp.iter()).for_each(|(x, y)| {
                        *x = op(*x, y);
                    });
                    if let Some(op3) = op3 {
                        res.iter_mut().for_each(|x| {
                            *x = op3(*x);
                        });
                    }
                });
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let outer_loop_size = result.size() / inner_loop_size;
            let iterators = uncontiguous_reduce::ReductionPreprocessor::new2(
                num_threads,
                outer_loop_size,
                inner_loop_size,
                a.ptr(),
                result.ptr(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                result.shape().clone(),
                result.strides().inner(),
            );
            let res_shape = result.shape().clone();
            iterators.into_par_iter().for_each(|mut iterator| {
                let mut result_ptr_c = iterator.res_ptrs.clone();
                let mut a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let res_last_strides = *result.strides().inner().last().unwrap();
                let res_strides = result.strides().clone();
                let res_shape = res_shape.clone();
                let shape_len = iterator.shape.len() as i64;
                for _i in 0..current_size {
                    for _ in 0..inner_loop_size_2 {
                        for i in 0..inner_loop_size as i64 {
                            result_ptr_c[i * res_last_strides] =
                                op(result_ptr_c[i * res_last_strides], a_data_ptr[i]);
                        }
                        for j in (shape_len..iterator.a_shape.len() as i64).rev() {
                            if iterator.prg[j as usize] < iterator.a_shape[j as usize] {
                                iterator.prg[j as usize] += 1;
                                a_data_ptr.offset(iterator.strides[j as usize]);
                                break;
                            } else {
                                iterator.prg[j as usize] = 0;
                                a_data_ptr.offset(
                                    -iterator.strides[j as usize] * iterator.a_shape[j as usize],
                                );
                            }
                        }
                    }
                    for j in (0..shape_len - 1).rev() {
                        if iterator.a_prg[j as usize] < iterator.a_shape[j as usize] {
                            iterator.a_prg[j as usize] += 1;
                            a_data_ptr.offset(iterator.strides[j as usize]);
                            break;
                        } else {
                            iterator.a_prg[j as usize] = 0;
                            a_data_ptr.offset(
                                -iterator.strides[j as usize] * iterator.a_shape[j as usize],
                            );
                        }
                    }
                    if let Some(op3) = op3 {
                        for i in 0..inner_loop_size as i64 {
                            result_ptr_c[i * res_last_strides] =
                                op3(result_ptr_c[i * res_last_strides]);
                        }
                    }
                    for j in (0..res_shape.len() - 1).rev() {
                        if iterator.res_prg[j] < res_shape[j] - 1 {
                            iterator.res_prg[j] += 1;
                            result_ptr_c.offset(res_strides[j]);
                            break;
                        } else {
                            iterator.res_prg[j] = 0;
                            result_ptr_c.offset(-res_strides[j] * (res_shape[j] - 1));
                        }
                    }
                    iterator.prg.iter_mut().for_each(|x| {
                        *x = 0;
                    });
                }
            });
        },
    )
}
