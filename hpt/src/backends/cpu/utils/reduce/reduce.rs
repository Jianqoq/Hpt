use crate::backends::cpu::kernels::argreduce_kernels::{argmax_kernel, argmin_kernel};
use crate::backends::cpu::utils::reduce::reduce_template::contiguous_reduce_template;
use crate::tensor_base::_Tensor;

use crate::backends::cpu::utils::reduce::reduce_utils::{
    ReductionPreprocessor, UCReductionPreprocessor,
};
use crate::THREAD_POOL;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape::Shape;
use hpt_common::shape::shape_utils::{mt_intervals, mt_intervals_simd};
use hpt_iterator::iterator_traits::StridedIterator;
use hpt_iterator::TensorIterator;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::shape_manipulate::ShapeManipulate;
use hpt_traits::ops::slice::Slice;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_traits::tensor::TensorLike;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::{Cmp, NormalOut};
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
use std::sync::Arc;
use std::sync::Barrier;

macro_rules! init_arr {
    (
        $result:ident,
        $shape:ident,
        $macro_init_val:expr,
        $($specific_type:tt)*
    ) => {
        $result = _Tensor::<$($specific_type)*, Cpu, DEVICE, Al>::empty($shape.clone())?;
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
        let a_: &_Tensor<$generic_a, Cpu, DEVICE, Al> = &$a;
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
        let a_data = a_.ptr::<T>();
        let mut res_shape = Vec::with_capacity(a_.ndim() - $axes.len());
        a_shape_cpy.iter().for_each(|x| {
            (if *x != 0 {
                res_shape.push(*x)
            })
        });
        let mut new_shape: Option<Vec<i64>> = None;
        let mut result;
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
            ShapeError::check_inplace_out_layout_valid(&Shape::from(res_shape.clone()), &out.layout())?;
            result = out;
            result_size = result.size();
        } else {
            init_arr!(result, res_shape, $init_val, $($specific_type)*);
            result_size = result.size();
        }
        let result_data = result.ptr::<$($specific_type)*>();
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
        #[track_caller]
        pub(crate) fn $fn_name<$generic_a, $generic_b, const DEVICE: usize, Al>(a: &_Tensor<$generic_a, Cpu, DEVICE, Al>, axes: Vec<usize>,
             init_val: $generic_b, keepdims: bool, c: Option<_Tensor<$generic_b, Cpu, DEVICE, Al>>) -> std::result::Result<_Tensor<$generic_b, Cpu, DEVICE, Al>, TensorError> $($trait_bound)*
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
        #[track_caller]
        pub(crate) fn $fn_name<$generic_a, const DEVICE: usize, Al>(a: &_Tensor<$generic_a, Cpu, DEVICE, Al>, axes: Vec<usize>,
             init_val: $generic_a, keepdims: bool, c: Option<_Tensor<$generic_a, Cpu, DEVICE, Al>>) -> std::result::Result<_Tensor<$generic_a, Cpu, DEVICE, Al>, TensorError> $($trait_bound)*
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
        #[track_caller]
        pub(crate) fn $fn_name<$generic_a, const DEVICE: usize, Al>(a: &_Tensor<$generic_a, Cpu, DEVICE, Al>, axes: Vec<usize>,
             init_val: $($specific_type)*, keepdims: bool, c: Option<_Tensor<$($specific_type)*, Cpu, DEVICE, Al>>) -> std::result::Result<_Tensor<$($specific_type)*, Cpu, DEVICE, Al>, TensorError> $($trait_bound)*
         {
            body_one_axis!(axes, a, init_val, keepdims, c, $kernel_name, $generic_a, $($specific_type)*);
        }
    };
}

use hpt_types::vectors::traits::*;

use super::reduce_template::uncontiguos_reduce_template;
use crate::backends::cpu::kernels::reduce::{
    contiguous_reduce_dim_include, contiguous_reduce_dim_include_simd,
    uncontiguous_reduce_dim_include,
};

#[track_caller]
pub(crate) fn reduce<T, F, F2, F3, F4, F5, F6, O, const DEVICE: usize, Al>(
    a: &_Tensor<T, Cpu, DEVICE, Al>,
    preop: F,
    preop_no_cast: F2,
    cumulate: F3,
    vec_preop: F4,
    vec_preop_no_cast: F5,
    vec_cumulate: F6,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O, Cpu, DEVICE, Al>>,
) -> std::result::Result<_Tensor<O, Cpu, DEVICE, Al>, TensorError>
where
    T: CommonBounds + Cast<O>,
    F: Fn(T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(T::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    F5: Fn(O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    F6: Fn(O::Vec, O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    O: CommonBounds,
    T::Vec: Copy,
    O::Vec: Copy,
    Al: Allocator + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
{
    if a.is_contiguous() && a.parent::<T>().is_none() {
        contiguous_reduce(
            a,
            preop,
            preop_no_cast,
            cumulate,
            None::<fn(O) -> O>,
            vec_preop,
            vec_preop_no_cast,
            vec_cumulate,
            None::<fn(O::Vec) -> O::Vec>,
            &axes,
            init_val,
            keepdims,
            init_out,
            c,
        )
    } else {
        uncontiguous_reduce(
            a,
            preop,
            cumulate,
            None::<fn(O) -> O>,
            &axes,
            init_val,
            keepdims,
            init_out,
            c,
        )
    }
}

#[track_caller]
pub(crate) fn reduce_with_post<T, F, F2, F3, F4, F5, F6, F7, F8, O, const DEVICE: usize, Al>(
    a: &_Tensor<T, Cpu, DEVICE, Al>,
    preop: F,
    preop_no_cast: F2,
    cumulate: F3,
    postop: F4,
    vec_preop: F5,
    vec_preop_no_cast: F6,
    vec_cumulate: F7,
    vec_postop: F8,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O, Cpu, DEVICE, Al>>,
) -> std::result::Result<_Tensor<O, Cpu, DEVICE, Al>, TensorError>
where
    T: CommonBounds + Cast<O>,
    F: Fn(T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(O) -> O + Sync + Send + 'static + Copy,
    F5: Fn(T::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    F6: Fn(O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    F7: Fn(O::Vec, O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    F8: Fn(O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    O: CommonBounds,
    O::Vec: Copy,
    Al: Allocator + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
{
    if a.is_contiguous() && a.parent::<T>().is_none() {
        contiguous_reduce(
            a,
            preop,
            preop_no_cast,
            cumulate,
            Some(postop),
            vec_preop,
            vec_preop_no_cast,
            vec_cumulate,
            Some(vec_postop),
            &axes,
            init_val,
            keepdims,
            init_out,
            c,
        )
    } else {
        uncontiguous_reduce(
            a,
            preop,
            cumulate,
            Some(postop),
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
    where T: CommonBounds + NormalOut<T, Output = T> + Cmp<T, Output = bool>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
);

register_reduction_one_axis!(
    T => [i64],
    argmin,
    argmin_kernel,
    where T: CommonBounds + NormalOut<T, Output = T> + Cmp<T, Output = bool>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
);

#[track_caller]
pub(crate) fn contiguous_reduce<T, F, F2, F3, F4, F5, F6, F7, F8, O, const DEVICE: usize, A>(
    a: &_Tensor<T, Cpu, DEVICE, A>,
    preop: F,
    preop_no_cast: F2,
    cumulate: F3,
    postop: Option<F4>,
    vec_preop: F5,
    vec_preop_no_cast: F6,
    vec_cumulate: F7,
    vec_postop: Option<F8>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O, Cpu, DEVICE, A>>,
) -> std::result::Result<_Tensor<O, Cpu, DEVICE, A>, TensorError>
where
    T: CommonBounds + Cast<O>,
    O: CommonBounds,
    F: Fn(T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F4: Fn(O) -> O + Sync + Send + 'static + Copy,
    F5: Fn(T::Vec) -> O::Vec + 'static + Copy + Send + std::marker::Sync,
    F6: Fn(O::Vec) -> O::Vec + 'static + Copy + Send + std::marker::Sync,
    F7: Fn(O::Vec, O::Vec) -> O::Vec + 'static + Copy + Send + std::marker::Sync,
    F8: Fn(O::Vec) -> O::Vec + Sync + Send + 'static + Copy,
    T::Vec: Copy,
    O::Vec: Copy,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    let res_shape = a.layout.reduce(axes, keepdims)?.shape().clone();
    let res = contiguous_reduce_template(
        &a,
        axes,
        init_val,
        keepdims,
        init_out,
        c,
        |res| {
            let ptr = a.ptr();
            let raw = unsafe { std::slice::from_raw_parts_mut(ptr.ptr, a.size() as usize) };
            let val = raw
                .par_iter()
                .fold(|| init_val, |acc, &x| cumulate(acc, preop(x)))
                .reduce(|| init_val, |a, b| cumulate(a, b));
            if let Some(postop) = postop {
                *res = postop(val);
            } else {
                *res = val;
            }
        },
        |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let iterators = ReductionPreprocessor::new(
                num_threads,
                result.size(),
                inner_loop_size_2,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                result.shape().clone(),
            );
            iterators.into_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.a_shape.len() as i64;
                if T::STR == O::STR {
                    contiguous_reduce_dim_include_simd(
                        init_val,
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr.cast::<O>(),
                        result_ptr_c,
                        &iterator.strides,
                        &iterator.a_shape,
                        &mut iterator.prg,
                        shape_len,
                        preop_no_cast,
                        cumulate,
                        vec_preop_no_cast,
                        vec_cumulate,
                        postop,
                    );
                } else {
                    contiguous_reduce_dim_include(
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr,
                        result_ptr_c,
                        &iterator.strides,
                        &iterator.a_shape,
                        &mut iterator.prg,
                        shape_len,
                        preop,
                        cumulate,
                        postop,
                    );
                }
            });
        },
        |num_threads, inner_loop_size, result, a| {
            let intervals = mt_intervals_simd(inner_loop_size, num_threads, O::Vec::SIZE);
            let mut slices = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); a.ndim()];
            let mut slices_res = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); result.ndim()];
            let mut sliced_tensors = Vec::with_capacity(num_threads);
            let mut sliced_res = Vec::with_capacity(num_threads);
            assert_eq!(inner_loop_size, result.size());
            for (start, end) in intervals.into_iter() {
                if end - start == 0 {
                    continue;
                }
                slices[a.ndim() - 1] = (start as i64, end as i64, 1);
                slices_res[result.ndim() - 1] = (start as i64, end as i64, 1);
                sliced_tensors.push(a.slice(&slices).expect("Slice failed"));
                sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
            }
            sliced_tensors
                .into_par_iter()
                .zip(sliced_res.into_par_iter())
                .for_each(move |(inp, res)| {
                    let inp_ptr = inp.ptr();
                    let res_ptr = res.ptr();

                    let inner_loop_size = *res.shape().last().unwrap() as isize;
                    let outer_loop_size = (inp.size() as isize) / inner_loop_size;
                    use crate::backends::cpu::kernels::reduce::fast_reduce_no_simd;
                    use crate::backends::cpu::kernels::reduce::fast_reduce_simd;
                    if O::Vec::SIZE == T::Vec::SIZE {
                        fast_reduce_simd(
                            inner_loop_size,
                            outer_loop_size,
                            inp_ptr,
                            res_ptr,
                            inp.strides().inner(),
                            inp.shape().inner(),
                            O::Vec::SIZE as isize,
                            preop,
                            cumulate,
                            postop,
                            vec_preop,
                            vec_cumulate,
                            vec_postop,
                        );
                    } else {
                        fast_reduce_no_simd(
                            inner_loop_size,
                            outer_loop_size,
                            inp_ptr,
                            res_ptr,
                            inp.strides().inner(),
                            inp.shape().inner(),
                            preop,
                            cumulate,
                            postop,
                        );
                    }
                });
        },
        |num_threads,
         outer_loop_size,
         inner_loop_size,
         inner_loop_size_2,
         result,
         transposed_tensor| {
            let iterators = ReductionPreprocessor::new2(
                num_threads,
                outer_loop_size,
                inner_loop_size,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                result.shape().clone(),
            );
            iterators.into_par_iter().for_each(|iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.shape.len() as i64;
                let inp_strides = &iterator.strides;
                let inp_shape = &iterator.a_shape;
                let mut prg1 = iterator.prg.clone();
                let mut prg2 = iterator.a_prg.clone();
                use crate::backends::cpu::kernels::reduce::reduce_dim_not_include;
                use crate::backends::cpu::kernels::reduce::reduce_dim_not_include_simd;
                if O::Vec::SIZE == T::Vec::SIZE {
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
                        preop,
                        cumulate,
                        postop,
                        vec_preop,
                        vec_cumulate,
                        vec_postop,
                    );
                } else {
                    reduce_dim_not_include(
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
                        preop,
                        cumulate,
                        postop,
                    );
                }
            });
        },
    )?;
    res.reshape(res_shape)
}

#[track_caller]
pub(crate) fn uncontiguous_reduce<T, F, F2, F3, O, const DEVICE: usize, Al>(
    a: &_Tensor<T, Cpu, DEVICE, Al>,
    preop: F,
    cumulate: F2,
    postop: Option<F3>,
    axes: &[usize],
    init_val: O,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<O, Cpu, DEVICE, Al>>,
) -> std::result::Result<_Tensor<O, Cpu, DEVICE, Al>, TensorError>
where
    T: CommonBounds + Cast<O>,
    O: CommonBounds,
    F: Fn(T) -> O + Sync + Send + 'static + Copy,
    F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
    F3: Fn(O) -> O + Sync + Send + 'static + Copy,
    T::Vec: Copy,
    O::Vec: Copy,
    Al: Allocator + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
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
                .par_iter()
                .par_strided_fold(init_val, |acc, x| cumulate(acc, preop(x)))
                .reduce(|| init_val, |a, b| cumulate(a, b));
            if let Some(postop) = postop {
                *res = postop(val);
            } else {
                *res = val;
            }
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let a_last_stride = transposed_tensor.strides()[a.ndim() - 1];
            let iterators = UCReductionPreprocessor::new(
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
                let result_ptr_c = iterator.res_ptrs;
                let a_data_ptr = iterator.ptrs;
                let current_size = iterator.end - iterator.start;
                let res_shape = res_shape.clone();
                let res_strides = result.strides().clone();
                let shape_len = iterator.a_shape.len() as i64;

                uncontiguous_reduce_dim_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    inner_loop_size_2 as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    &mut iterator.res_prg,
                    &res_strides,
                    &res_shape,
                    shape_len,
                    a_last_stride as isize,
                    preop,
                    cumulate,
                    postop,
                );
            });
        },
        move |num_threads, inner_loop_size, ap, result| {
            let intervals = mt_intervals(inner_loop_size, num_threads);
            let mut slices = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); ap.ndim()];
            let mut slices_res = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); result.ndim()];
            let mut sliced_tensors = Vec::with_capacity(num_threads);
            let mut sliced_res = Vec::with_capacity(num_threads);
            assert_eq!(inner_loop_size, result.size());
            for (start, end) in intervals.into_iter() {
                if end - start == 0 {
                    continue;
                }
                slices[ap.ndim() - 1] = (start as i64, end as i64, 1);
                slices_res[result.ndim() - 1] = (start as i64, end as i64, 1);
                sliced_tensors.push(ap.slice(&slices).expect("Slice failed"));
                sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
            }
            sliced_tensors
                .into_par_iter()
                .zip(sliced_res.into_par_iter())
                .for_each(move |(inp, mut res)| {
                    res.iter_mut().zip(inp.iter()).for_each(|(x, y)| {
                        *x = cumulate(*x, preop(y));
                    });
                    if let Some(postop) = postop {
                        res.iter_mut().for_each(|x| {
                            *x = postop(*x);
                        });
                    }
                });
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let outer_loop_size = result.size() / inner_loop_size;
            let iterators = UCReductionPreprocessor::new2(
                num_threads,
                outer_loop_size,
                inner_loop_size,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                result.shape().clone(),
                result.strides().inner(),
            );
            let res_shape = result.shape().clone();
            iterators.into_par_iter().for_each(|mut iterator| {
                let a_last_stride = transposed_tensor.strides()[a.ndim() - axes.len() - 1];
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let res_last_strides = *result.strides().inner().last().unwrap();
                let res_strides = result.strides().clone();
                let res_shape = res_shape.clone();
                let shape_len = iterator.shape.len() as i64;
                use crate::backends::cpu::kernels::reduce::uncontiguous_reduce_dim_not_include;
                uncontiguous_reduce_dim_not_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    inner_loop_size_2 as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    &mut iterator.a_prg,
                    &mut iterator.res_prg,
                    &res_strides,
                    &res_shape,
                    shape_len,
                    a_last_stride as isize,
                    res_last_strides as isize,
                    preop,
                    cumulate,
                    postop,
                );
            });
        },
    )
}
