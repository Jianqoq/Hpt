use crate::slice::SliceOps;
use crate::tensor_base::_Tensor;
use crate::{ argmax_kernel, argmin_kernel };
use crate::backend::Cpu;

use tensor_common::slice::Slice;
use tensor_common::axis::{ process_axes, Axis };
#[cfg(feature = "simd")]
use tensor_iterator::iterator_traits::StridedIteratorSimd;
use tensor_traits::TensorLike;
use tensor_types::into_scalar::IntoScalar;
use rayon::iter::IntoParallelRefIterator;
use tensor_types::dtype::TypeCommon;
use tensor_traits::tensor::{ FloatReduce, IndexReduce, NormalReduce, TensorInfo };
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::borrow::BorrowMut;
use tensor_traits::tensor::TensorCreator;
use crate::THREAD_POOL;
use tensor_common::shape_utils::{ mt_intervals, predict_reduce_shape };
use tensor_common::pointer::Pointer;
use anyhow;
use tensor_traits::tensor::CommonBounds;
use tensor_types::convertion::FromScalar;
use tensor_types::type_promote::{ Cmp, Eval, FloatOut, NormalOut };
use std::sync::Arc;
use std::sync::Barrier;
use tensor_traits::shape_manipulate::ShapeManipulate;

use super::unary::FloatType;

#[derive(Debug)]
struct ReductionPreprocessor<T, U> {
    pub ptrs: Pointer<T>,
    pub res_ptrs: Pointer<U>,
    pub strides: Vec<i64>,
    pub start: usize,
    pub end: usize,
    pub prg: Vec<i64>,
    pub a_prg: Vec<i64>,
    pub shape: Arc<Vec<i64>>,
    pub a_shape: Arc<Vec<i64>>,
}

impl<T, U> ReductionPreprocessor<T, U> where T: Clone, U: Clone {
    pub fn new(
        num_threads: usize,
        loop_size: usize,
        inner_loop_size: usize,
        ptrs: Pointer<T>,
        mut res_ptrs: Pointer<U>,
        strides: Vec<i64>,
        a_shape: Arc<Vec<i64>>,
        transposed_shape: Arc<Vec<i64>>,
        res_shape: Arc<Vec<i64>>
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
            let mut prg =
                vec![0; a_shape.len() - 1]; /* -1 because we want to escape the last axis */

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
        transposed_strides: Vec<i64>,
        transposed_shape: Arc<Vec<i64>>,
        res_shape: Arc<Vec<i64>>
    ) -> Vec<ReductionPreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; res_shape.len()];
        let res_ptrs = res_ptrs.borrow_mut();
        let ndim = res_shape.len() as i64;
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();

            for i in (0..=ndim - 2).rev() {
                a_data_ptr_cpy.offset(
                    progress_init_a_data[i as usize] * transposed_strides[i as usize]
                );
            }

            let progress_init_a_data_cpy = progress_init_a_data.clone();

            task_amout += intervals[id].1 - intervals[id].0;

            let prg = vec![0; transposed_shape.len()];

            let res_ptr_cpy = res_ptrs.clone();
            res_ptrs.add((intervals[id].1 - intervals[id].0) * inner_loop_size);

            let mut tmp = task_amout as i64;
            for j in (0..=ndim - 2).rev() {
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

    pub fn reset_prg(&mut self) {
        self.prg.iter_mut().for_each(|x| {
            *x = 0;
        });
    }
}

#[allow(unused_macros)]
macro_rules! register_stack {
    ($name:ident) => {
        #[cfg_attr(feature = "track_caller", track_caller)]
        pub(crate) fn $name<T>(
            tensors: Vec<&_Tensor<T>>,
            axis: usize,
            keepdims: bool,
        ) -> anyhow::Result<_Tensor<T>>
        where
            T: CommonBounds,
        {
            let length = tensors.len();
            let mut all_same_shape = true;
            for i in tensors.iter() {
                tensors[0]
                    .shape()
                    .iter()
                    .enumerate()
                    .try_for_each(|(idx, x)| {
                        if idx != axis
                            && i.shape().len() == tensors[0].shape().len()
                            && *x != i.shape()[idx]
                        {
                            return Err(anyhow::Error::msg(
                                "Shapes except the axis to stack must be the same",
                            ));
                        } else if i.shape().len() != tensors[0].shape().len() {
                            return Err(anyhow::Error::msg(
                                "Shape length mismatch when trying to stack tensors",
                            ));
                        } else if idx == axis && *x != i.shape()[idx] {
                            all_same_shape = false;
                        }
                        return Ok(());
                    })?;
            }
            let mut new_shape: Vec<usize> = vec![0; tensors[0].ndim() as usize];
            tensors.iter().for_each(|x| {
                new_shape[axis] += x.shape()[axis];
            });
            tensors[0].shape().iter().enumerate().for_each(|(i, x)| {
                if i != axis {
                    new_shape[i] = *x;
                }
            });
            let new_tensor: _Tensor<T> = _Tensor::empty(&new_shape)?;
            let mut res_ptr;
            let mut res_ptr_cpy = new_tensor.ptr();

            // total size to handle the tensors for each slot
            //_Tensor([[[ 0  2]
            //         [ 1  3]
            //         [ 0  2]
            //         [ 1  3]]
            //        [[ 4  6]
            //         [ 5  7]
            //         [ 4  6]
            //         [ 5  7]]])
            // here [0 2 1 3] is one slo for one tensor, since we have 2 tensors, we need 2 slots, total size is 8
            let mut total = 0;
            for tensor in &tensors {
                let mut jump = 1;
                for i in axis..new_shape.len() {
                    jump *= tensor.shape()[i];
                }
                total += jump;
            }
            for tensor in tensors {
                res_ptr = res_ptr_cpy.clone();
                let a_data = tensor.ptr();
                let a_strides = tensor.strides().as_ref().clone();
                let a_last_stride = a_strides[(tensor.ndim() - 1) as usize];
                let mut a_shape_cpy = tensor.shape().as_ref().clone();
                let mut offset = Vec::with_capacity(tensor.ndim() as usize);

                // calculate slot size for current tensor
                let mut current_slot_size = 1;
                for i in axis..tensor.shape().len() {
                    current_slot_size *= tensor.shape()[i];
                }
                let inner_loop_size = tensor.shape()[(tensor.ndim() - 1) as usize];
                let outer_loop_size = tensor.size() / inner_loop_size;
                for i in 0..tensor.ndim() as usize {
                    a_shape_cpy[i] -= 1;
                    offset.push(a_shape_cpy[i] * a_strides[i]);
                }
                let t = tensor.shape()[tensor.shape().len() - 1];
                let a_shape_arc = Arc::new(tensor.shape().as_ref().clone());
                let num_threads;
                unsafe {
                    if outer_loop_size < THREAD_POOL.max_count() {
                        num_threads = outer_loop_size;
                    } else {
                        num_threads = THREAD_POOL.max_count();
                    }
                }
                let mut iterators = StackPreprocessor::stack_new(
                    num_threads,
                    outer_loop_size,
                    inner_loop_size,
                    current_slot_size,
                    total,
                    t,
                    vec![a_data],
                    vec![res_ptr],
                    vec![a_strides],
                    vec![offset],
                    a_shape_arc.clone(),
                );
                let barrier = Arc::new(Barrier::new(num_threads + 1));

                THREAD_POOL.with(|pool| {
                    for _ in 0..num_threads {
                        let tmp = iterators.pop().unwrap();
                        let mut iterator = tmp.0;
                        let mut res_ptrs = iterator.res_ptrs[0];
                        let mut a_ptr = iterator.ptrs[0];
                        let current_size = iterator.end - iterator.start;
                        let barrier_clone = Arc::clone(&barrier);
                        let mut res_prg = tmp.1;
                        unsafe {
                            pool.execute(move || {
                                let a_strides = &iterator.strides[0];
                                let a_indices_cache = &iterator.offsets[0];
                                for _ in 0..current_size {
                                    for i in 0..inner_loop_size {
                                        let value = a_ptr[i * a_last_stride];
                                        res_ptrs.modify(i, value);
                                    }
                                    if res_prg + t < current_slot_size {
                                        res_ptrs.add(t);
                                        res_prg += t;
                                    } else {
                                        res_ptrs.add(total + t - current_slot_size);
                                        res_prg = 0;
                                    }
                                    for j in (0..=iterator.shape.len() as i64 - 2).rev() {
                                        if iterator.prg[j as usize] < iterator.shape[j as usize] - 1
                                        {
                                            iterator.prg[j as usize] += 1;
                                            a_ptr.add(a_strides[j as usize]);
                                            break;
                                        } else {
                                            iterator.prg[j as usize] = 0;
                                            a_ptr.sub(a_indices_cache[j as usize]);
                                        }
                                    }
                                }
                                barrier_clone.wait();
                            });
                        }
                    }
                });
                barrier.wait();
                res_ptr_cpy.add(current_slot_size);
            }
            if keepdims {
                if !all_same_shape {
                    return Err(anyhow::Error::msg(
                        "keepdims is not supported for different shapes",
                    ));
                }
                let mut res_shape = Vec::with_capacity(new_shape.len() + 1);
                for (idx, i) in new_shape.iter().enumerate() {
                    if idx == axis {
                        res_shape.push(length);
                        res_shape.push(*i / length);
                    } else {
                        res_shape.push(*i);
                    }
                }
                return new_tensor.reshape(res_shape);
            } else {
                return Ok(new_tensor);
            }
        }
    };
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
        let transposed_strides = transposed_tensor.strides().inner();
        let transposed_strides_cpy = transposed_strides.clone();
        let transposed_shape = transposed_tensor.shape().to_vec();
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
        let transposed_shape = Arc::new(transposed_shape);
        let a_last_index = a_.ndim() - 1;
        let inner_loop_size = transposed_shape[a_last_index];
        let a_data_ptr = a_data.clone();
        let last_stride = transposed_strides_cpy[a_last_index];
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
                transposed_strides_cpy,
                Arc::new(transposed_shape_cpy),
                transposed_shape.clone(),
                res_shape.clone(),
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

// #[cfg(not(feature = "simd"))]
// #[cfg_attr(feature = "track_caller", track_caller)]
// pub(crate) fn _reduce<T, F, F2, F3, O>(
//     a: &_Tensor<T>,
//     op: F,
//     op2: F2,
//     op3: Option<F3>,
//     axes: &[usize],
//     init_val: O,
//     keepdims: bool,
//     init_out: bool,
//     c: Option<_Tensor<O>>
// )
//     -> anyhow::Result<_Tensor<O>>
//     where
//         T: CommonBounds,
//         O: CommonBounds,
//         F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
//         F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
//         F3: Fn(O) -> O + Sync + Send + 'static + Copy
// {
//     use tensor_iterator::iterator_traits::StridedIterator;

//     let mut is_left: bool = true;
//     for axis in axes.iter() {
//         if axis == &((a.ndim() as usize) - 1) {
//             is_left = false;
//             break;
//         }
//     }
//     let a_: &_Tensor<T> = &a;
//     let a_shape = a_.shape();
//     let a_last_stride = a_.strides()[a_.ndim() - 1];
//     let a_shape_tmp = a_shape.clone();
//     let (a_shape_cpy, res_shape) = predict_reduce_shape(&a_shape_tmp, &axes);
//     let mut j = a_.ndim() - axes.len();
//     let mut k = 0;
//     let mut track_idx = 0;
//     let mut transposed_axis = vec![0; a_.ndim()];
//     for i in 0..a_.ndim() {
//         if a_shape_cpy[i] != 0 {
//             transposed_axis[k] = i;
//             k += 1;
//         } else {
//             transposed_axis[j] = axes[track_idx];
//             j += 1;
//             track_idx += 1;
//         }
//     }
//     transposed_axis[a.ndim() - axes.len()..].sort();
//     transposed_axis[..a.ndim() - axes.len()].sort();
//     let transposed_tensor = a_.permute(transposed_axis)?;
//     let transposed_strides = transposed_tensor.strides().inner();
//     let transposed_strides_cpy = transposed_strides.clone();
//     let transposed_shape = transposed_tensor.shape().to_vec();
//     let mut transposed_shape_cpy = transposed_shape.clone();
//     transposed_shape_cpy.iter_mut().for_each(|x| {
//         *x -= 1;
//     });
//     let a_data: Pointer<T> = a_.ptr();
//     let mut new_shape: Option<Vec<i64>> = None;
//     let result;
//     let result_size: usize;
//     if keepdims {
//         let mut shape_tmp = Vec::with_capacity(a_.ndim());
//         a_shape_cpy.iter().for_each(|x| {
//             if *x != 0 {
//                 shape_tmp.push(*x);
//             } else {
//                 shape_tmp.push(1);
//             }
//         });
//         new_shape = Some(shape_tmp);
//     }
//     let res_shape = Arc::new(res_shape);
//     if let Some(out) = c {
//         if let Some(s) = &new_shape {
//             if s != out.shape().inner() {
//                 return Err(anyhow::Error::msg(format!("Output array has incorrect shape")));
//             }
//         } else {
//             if res_shape.as_ref() != out.shape().inner() {
//                 return Err(anyhow::Error::msg(format!("Output array has incorrect shape")));
//             }
//         }
//         result = out;
//         result_size = result.size();
//         if init_out {
//             result
//                 .as_raw_mut()
//                 .par_iter_mut()
//                 .for_each(|x| {
//                     *x = init_val;
//                 });
//         }
//     } else {
//         result = _Tensor::<O, Cpu>::empty(res_shape.clone())?;
//         result
//             .as_raw_mut()
//             .par_iter_mut()
//             .for_each(|x| {
//                 *x = init_val;
//             });
//         result_size = result.size();
//     }
//     let mut result_data = result.ptr();
//     let transposed_shape: Arc<Vec<i64>> = Arc::new(transposed_shape);
//     if a_.ndim() == axes.len() {
//         let val = a_
//             .as_raw_mut()
//             .par_iter()
//             .fold(
//                 || init_val,
//                 |acc, &x| op(acc, x)
//             )
//             .reduce(
//                 || init_val,
//                 |a, b| op2(a, b)
//             );
//         if let Some(op3) = op3 {
//             result_data.write(op3(op2(val, result_data.read())));
//         } else {
//             result_data.write(op2(val, result_data.read()));
//         }
//     } else {
//         let a_last_index: usize = a_.ndim() - 1;
//         let inner_loop_size: usize = a_.shape()[a_last_index] as usize;
//         let a_size: usize = a_.size();
//         let a_data_ptr: Pointer<T> = a_data.clone();
//         THREAD_POOL.with_borrow_mut(|pool| {
//             if !is_left {
//                 let outer_loop_size = a_size / inner_loop_size;
//                 let inner_loop_size_2 = outer_loop_size / result_size;
//                 let num_threads;
//                 if result_size < pool.max_count() {
//                     num_threads = result_size;
//                 } else {
//                     num_threads = pool.max_count();
//                 }
//                 let mut iterators = ReductionPreprocessor::new(
//                     num_threads,
//                     result_size,
//                     inner_loop_size_2,
//                     a_data_ptr,
//                     result_data,
//                     transposed_strides_cpy,
//                     Arc::new(transposed_shape_cpy),
//                     transposed_shape.clone(),
//                     res_shape.clone()
//                 );
//                 let barrier = Arc::new(Barrier::new(num_threads + 1));
//                 for _ in 0..num_threads {
//                     let mut iterator = iterators.pop().unwrap();
//                     let mut result_ptr_c = iterator.res_ptrs;
//                     let mut a_data_ptr = iterator.ptrs;
//                     let current_size = iterator.end - iterator.start;
//                     let barrier_clone = Arc::clone(&barrier);
//                     pool.execute(move || {
//                         let shape_len = iterator.a_shape.len() as i64;
//                         for _ in 0..current_size {
//                             for _ in 0..inner_loop_size_2 {
//                                 let mut tmp = result_ptr_c[0isize];
//                                 for i in 0..inner_loop_size as i64 {
//                                     let a_val = a_data_ptr[i * a_last_stride];
//                                     tmp = op(tmp, a_val);
//                                 }
//                                 result_ptr_c[0isize] = tmp;
//                                 for j in (0..shape_len - 1).rev() {
//                                     if iterator.prg[j as usize] < iterator.a_shape[j as usize] {
//                                         iterator.prg[j as usize] += 1;
//                                         a_data_ptr.offset(iterator.strides[j as usize]);
//                                         break;
//                                     } else {
//                                         iterator.prg[j as usize] = 0;
//                                         a_data_ptr.offset(
//                                             -iterator.strides[j as usize] *
//                                                 iterator.a_shape[j as usize]
//                                         );
//                                     }
//                                 }
//                             }
//                             if let Some(op3) = op3 {
//                                 let tmp = result_ptr_c[0isize];
//                                 let tmp = op3(tmp);
//                                 result_ptr_c[0isize] = tmp;
//                             }
//                             result_ptr_c.add(1);
//                         }
//                         barrier_clone.wait();
//                     });
//                 }
//                 barrier.wait();
//             } else {
//                 let outer_loop_size = result_size / inner_loop_size;
//                 let inner_loop_size_2 = a.size() / result_size;
//                 if outer_loop_size == 1 {
//                     let num_threads = if inner_loop_size < pool.max_count() {
//                         inner_loop_size
//                     } else {
//                         pool.max_count()
//                     };
//                     let intervals = mt_intervals(inner_loop_size, num_threads);
//                     let mut slices = vec![Slice::Full; a.ndim() as usize];
//                     let mut slices_res = vec![Slice::Full; result.ndim() as usize];
//                     let mut sliced_tensors = Vec::with_capacity(num_threads);
//                     let mut sliced_res = Vec::with_capacity(num_threads);
//                     assert_eq!(inner_loop_size, result_size);
//                     for i in 0..num_threads {
//                         slices[(a.ndim() as usize) - 1] = Slice::Range((
//                             intervals[i].0 as i64,
//                             intervals[i].1 as i64,
//                         ));
//                         slices_res[(result.ndim() as usize) - 1] = Slice::Range((
//                             intervals[i].0 as i64,
//                             intervals[i].1 as i64,
//                         ));
//                         sliced_tensors.push(a.slice(&slices).expect("Slice failed"));
//                         sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
//                     }
//                     let barrier = Arc::new(Barrier::new(num_threads + 1));
//                     for (inp, res) in sliced_tensors.into_iter().zip(sliced_res.into_iter()) {
//                         let barrier_clone = barrier.clone();
//                         pool.execute(move || {
//                             res.iter_mut()
//                                 .zip(inp.iter())
//                                 .for_each(|(x, y)| {
//                                     *x = op(*x, y);
//                                 });
//                             if let Some(op3) = op3 {
//                                 res.iter_mut().for_each(|x| {
//                                     *x = op3(*x);
//                                 });
//                             }
//                             barrier_clone.wait();
//                         });
//                     }
//                     barrier.wait();
//                 } else {
//                     let num_threads = if outer_loop_size < pool.max_count() {
//                         outer_loop_size
//                     } else {
//                         pool.max_count()
//                     };
//                     let mut iterators = ReductionPreprocessor::new2(
//                         num_threads,
//                         outer_loop_size,
//                         inner_loop_size,
//                         a_data_ptr,
//                         result_data,
//                         transposed_strides_cpy,
//                         Arc::new(transposed_shape_cpy),
//                         res_shape.clone()
//                     );
//                     let barrier = Arc::new(Barrier::new(num_threads + 1));
//                     for _ in (0..num_threads).rev() {
//                         let mut iterator = iterators.pop().unwrap();
//                         let mut result_ptr_c = iterator.res_ptrs;
//                         let mut a_data_ptr = iterator.ptrs;
//                         let current_size = iterator.end - iterator.start;
//                         let barrier_clone = Arc::clone(&barrier);
//                         pool.execute(move || {
//                             let shape_len = iterator.shape.len() as i64;
//                             for _i in 0..current_size {
//                                 for _ in 0..inner_loop_size_2 {
//                                     for i in 0..inner_loop_size as i64 {
//                                         let a_val = a_data_ptr[i * a_last_stride];
//                                         let result_val = result_ptr_c[i];
//                                         let mut_ref = unsafe {
//                                             &mut *result_ptr_c.ptr.offset(i as isize)
//                                         };
//                                         *mut_ref = op(result_val, a_val);
//                                     }
//                                     for j in (shape_len..=(iterator.a_shape.len() as i64) -
//                                         1).rev() {
//                                         if iterator.prg[j as usize] < iterator.a_shape[j as usize] {
//                                             iterator.prg[j as usize] += 1;
//                                             a_data_ptr.offset(iterator.strides[j as usize]);
//                                             break;
//                                         } else {
//                                             iterator.prg[j as usize] = 0;
//                                             a_data_ptr.offset(
//                                                 -iterator.strides[j as usize] *
//                                                     iterator.a_shape[j as usize]
//                                             );
//                                         }
//                                     }
//                                 }
//                                 for j in (0..shape_len - 1).rev() {
//                                     if iterator.a_prg[j as usize] < iterator.a_shape[j as usize] {
//                                         iterator.a_prg[j as usize] += 1;
//                                         a_data_ptr.offset(iterator.strides[j as usize]);
//                                         break;
//                                     } else {
//                                         iterator.a_prg[j as usize] = 0;
//                                         a_data_ptr.offset(
//                                             -iterator.strides[j as usize] *
//                                                 iterator.a_shape[j as usize]
//                                         );
//                                     }
//                                 }
//                                 if let Some(op3) = op3 {
//                                     for i in 0..inner_loop_size as i64 {
//                                         let result_val = result_ptr_c[i];
//                                         let mut_ref = unsafe {
//                                             &mut *result_ptr_c.ptr.offset(i as isize)
//                                         };
//                                         *mut_ref = op3(result_val);
//                                     }
//                                 }
//                                 result_ptr_c.add(inner_loop_size);
//                                 iterator.reset_prg();
//                             }
//                             barrier_clone.wait();
//                         });
//                     }
//                     barrier.wait();
//                 }
//             }
//         });
//     }
//     if let Some(new_shape) = new_shape {
//         let result = result.reshape(new_shape)?;
//         Ok(result)
//     } else {
//         Ok(result)
//     }
// }

use tensor_types::into_vec::IntoVec;
use tensor_types::vectors::traits::*;

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn _reduce<T, F, F2, F3, F4, F5, O>(
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
    c: Option<_Tensor<O>>
)
    -> anyhow::Result<_Tensor<O>>
    where
        T: CommonBounds + IntoScalar<O>,
        O: CommonBounds,
        F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
        F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
        F3: Fn(O) -> O + Sync + Send + 'static + Copy,
        F4: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec +
            'static +
            Copy +
            std::marker::Send,
        F5: Fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec + Sync + Send + 'static + Copy,
        <T as TypeCommon>::Vec: IntoVec<<O as TypeCommon>::Vec> + Copy,
        <O as TypeCommon>::Vec: Copy
{
    use tensor_common::shape_utils::mt_intervals_simd;
    use tensor_iterator::iterator_traits::StridedIterator;

    let mut is_left: bool = true;
    for axis in axes.iter() {
        if axis == &((a.ndim() as usize) - 1) {
            is_left = false;
            break;
        }
    }
    let a_: &_Tensor<T> = &a;
    let a_shape = a_.shape();
    let a_last_stride = a_.strides()[a_.ndim() - 1];
    let a_shape_tmp = a_shape.clone();
    let (a_shape_cpy, res_shape) = predict_reduce_shape(&a_shape_tmp, &axes);
    let mut j = a_.ndim() - axes.len();
    let mut k = 0;
    let mut track_idx = 0;
    let mut transposed_axis = vec![0; a_.ndim()];
    for i in 0..a_.ndim() {
        if a_shape_cpy[i] != 0 {
            transposed_axis[k] = i;
            k += 1;
        } else {
            transposed_axis[j] = axes[track_idx];
            j += 1;
            track_idx += 1;
        }
    }
    transposed_axis[a.ndim() - axes.len()..].sort();
    transposed_axis[..a.ndim() - axes.len()].sort();
    let transposed_tensor = a_.permute(transposed_axis)?;
    let transposed_strides = transposed_tensor.strides().inner();
    let transposed_strides_cpy = transposed_strides.clone();
    let transposed_shape = transposed_tensor.shape().to_vec();
    let mut transposed_shape_cpy = transposed_shape.clone();
    transposed_shape_cpy.iter_mut().for_each(|x| {
        *x -= 1;
    });
    let a_data: Pointer<T> = a_.ptr();
    let mut new_shape: Option<Vec<i64>> = None;
    let result;
    let result_size: usize;
    if keepdims {
        let mut shape_tmp = Vec::with_capacity(a_.ndim());
        a_shape_cpy.iter().for_each(|x| {
            if *x != 0 {
                shape_tmp.push(*x);
            } else {
                shape_tmp.push(1);
            }
        });
        new_shape = Some(shape_tmp);
    }
    let res_shape = Arc::new(res_shape);
    if let Some(out) = c {
        if let Some(s) = &new_shape {
            if s != out.shape().inner() {
                return Err(anyhow::Error::msg(format!("Output array has incorrect shape")));
            }
        } else {
            if res_shape.as_ref() != out.shape().inner() {
                return Err(anyhow::Error::msg(format!("Output array has incorrect shape")));
            }
        }
        result = out;
        result_size = result.size();
        if init_out {
            result
                .as_raw_mut()
                .par_iter_mut()
                .for_each(|x| {
                    *x = init_val;
                });
        }
    } else {
        result = _Tensor::<O, Cpu>::empty(res_shape.clone())?;
        result
            .as_raw_mut()
            .par_iter_mut()
            .for_each(|x| {
                *x = init_val;
            });
        result_size = result.size();
    }
    let mut result_data = result.ptr();
    let transposed_shape: Arc<Vec<i64>> = Arc::new(transposed_shape);
    if a_.ndim() == axes.len() {
        let val = a_
            .as_raw_mut()
            .par_iter()
            .fold(
                || init_val,
                |acc, &x| op(acc, x)
            )
            .reduce(
                || init_val,
                |a, b| op2(a, b)
            );
        if let Some(op3) = op3 {
            result_data.write(op3(op2(val, result_data.read())));
        } else {
            result_data.write(op2(val, result_data.read()));
        }
    } else {
        let a_last_index: usize = a_.ndim() - 1;
        let inner_loop_size: usize = a_.shape()[a_last_index] as usize;
        let a_size: usize = a_.size();
        let a_data_ptr: Pointer<T> = a_data.clone();
        THREAD_POOL.with_borrow_mut(|pool| {
            if !is_left {
                let outer_loop_size = a_size / inner_loop_size;
                let inner_loop_size_2 = outer_loop_size / result_size;
                let num_threads;
                if result_size < pool.max_count() {
                    num_threads = result_size;
                } else {
                    num_threads = pool.max_count();
                }
                let mut iterators = ReductionPreprocessor::new(
                    num_threads,
                    result_size,
                    inner_loop_size_2,
                    a_data_ptr,
                    result_data,
                    transposed_strides_cpy,
                    Arc::new(transposed_shape_cpy),
                    transposed_shape.clone(),
                    res_shape.clone()
                );
                let barrier = Arc::new(Barrier::new(num_threads + 1));
                for _ in 0..num_threads {
                    let mut iterator = iterators.pop().unwrap();
                    let mut result_ptr_c = iterator.res_ptrs;
                    let mut a_data_ptr = iterator.ptrs;
                    let current_size = iterator.end - iterator.start;
                    let barrier_clone = Arc::clone(&barrier);
                    pool.execute(move || {
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
                                            -iterator.strides[j as usize] *
                                                iterator.a_shape[j as usize]
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
                        barrier_clone.wait();
                    });
                }
                barrier.wait();
            } else {
                let outer_loop_size = result_size / inner_loop_size;
                let inner_loop_size_2 = a.size() / result_size;
                if outer_loop_size == 1 {
                    let num_threads = if inner_loop_size < pool.max_count() {
                        inner_loop_size
                    } else {
                        pool.max_count()
                    };
                    let intervals = mt_intervals_simd(
                        inner_loop_size,
                        num_threads,
                        <O as TypeCommon>::Vec::SIZE
                    );
                    let mut slices = vec![Slice::Full; a.ndim() as usize];
                    let mut slices_res = vec![Slice::Full; result.ndim() as usize];
                    let mut sliced_tensors = Vec::with_capacity(num_threads);
                    let mut sliced_res = Vec::with_capacity(num_threads);
                    let mut num_threads = 0;
                    assert_eq!(inner_loop_size, result_size);
                    for (start, end) in intervals.into_iter() {
                        if end - start == 0 {
                            continue;
                        }
                        num_threads += 1;
                        slices[(a.ndim() as usize) - 1] = Slice::Range((start as i64, end as i64));
                        slices_res[(result.ndim() as usize) - 1] = Slice::Range((
                            start as i64,
                            end as i64,
                        ));
                        sliced_tensors.push(a.slice(&slices).expect("Slice failed"));
                        sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
                    }
                    let barrier = Arc::new(Barrier::new(num_threads + 1));
                    for (inp, res) in sliced_tensors.into_iter().zip(sliced_res.into_iter()) {
                        let barrier_clone = barrier.clone();
                        pool.execute(move || {
                            #[cfg(feature = "simd")]
                            res.iter_mut_simd()
                                .zip(inp.iter_simd())
                                .for_each(
                                    |(x, y)| {
                                        *x = op(*x, y);
                                    },
                                    |(x, y)| {
                                        *x = vec_op(*x, y);
                                    }
                                );
                            #[cfg(not(feature = "simd"))]
                            res.iter_mut()
                                .zip(inp.iter())
                                .for_each(|(x, y)| {
                                    *x = op(*x, y);
                                });
                            if let Some(op3) = op3 {
                                #[cfg(feature = "simd")]
                                res.iter_mut().for_each(|x| {
                                    *x = op3(*x);
                                });

                                #[cfg(not(feature = "simd"))]
                                {
                                    let op5 = vec_post.unwrap();
                                    res.iter_mut().for_each(|x| {
                                        *x = op3(*x);
                                    });
                                }
                            }
                            barrier_clone.wait();
                        });
                    }
                    barrier.wait();
                } else {
                    let num_threads = if outer_loop_size < pool.max_count() {
                        outer_loop_size
                    } else {
                        pool.max_count()
                    };
                    let mut iterators = ReductionPreprocessor::new2(
                        num_threads,
                        outer_loop_size,
                        inner_loop_size,
                        a_data_ptr,
                        result_data,
                        transposed_strides_cpy,
                        Arc::new(transposed_shape_cpy),
                        res_shape.clone()
                    );
                    let barrier = Arc::new(Barrier::new(num_threads + 1));
                    for _ in (0..num_threads).rev() {
                        let mut iterator = iterators.pop().unwrap();
                        let mut result_ptr_c = iterator.res_ptrs;
                        let mut a_data_ptr = iterator.ptrs;
                        let current_size = iterator.end - iterator.start;
                        let barrier_clone = Arc::clone(&barrier);
                        pool.execute(move || {
                            let shape_len = iterator.shape.len() as i64;

                            let vec_size = <T as TypeCommon>::Vec::SIZE;

                            let next1 = |
                                iterator: &mut ReductionPreprocessor<T, O>,
                                inp_ptr: &mut Pointer<T>
                            | {
                                for j in (shape_len..iterator.a_shape.len() as i64).rev() {
                                    let j = j as usize;
                                    if iterator.prg[j] < iterator.a_shape[j] {
                                        iterator.prg[j] += 1;
                                        inp_ptr.offset(iterator.strides[j]);
                                        break;
                                    } else {
                                        iterator.prg[j] = 0;
                                        inp_ptr.offset(-iterator.strides[j] * iterator.a_shape[j]);
                                    }
                                }
                            };

                            let next2 = |
                                iterator: &mut ReductionPreprocessor<T, O>,
                                inp_ptr: &mut Pointer<T>
                            | {
                                for j in (0..shape_len - 1).rev() {
                                    let j = j as usize;
                                    if iterator.a_prg[j] < iterator.a_shape[j] {
                                        iterator.a_prg[j] += 1;
                                        inp_ptr.offset(iterator.strides[j]);
                                        break;
                                    } else {
                                        iterator.a_prg[j] = 0;
                                        inp_ptr.offset(-iterator.strides[j] * iterator.a_shape[j]);
                                    }
                                }
                            };

                            #[cfg(feature = "simd")]
                            if a_last_stride == 1 {
                                let remain = inner_loop_size % vec_size;
                                let inner = inner_loop_size - remain;
                                for _i in 0..current_size {
                                    for _ in 0..inner_loop_size_2 {
                                        for i in 0..inner / vec_size {
                                            let a_vec = unsafe {
                                                <T as TypeCommon>::Vec::from_ptr(
                                                    a_data_ptr.ptr.add(i * vec_size)
                                                )
                                            };
                                            let result_vec = unsafe {
                                                <O as TypeCommon>::Vec::from_ptr(
                                                    result_ptr_c.ptr.add(i * vec_size)
                                                )
                                            };
                                            let res = vec_op(result_vec, a_vec);
                                            let ptr = res.as_ptr();
                                            unsafe {
                                                std::ptr::copy_nonoverlapping(
                                                    ptr,
                                                    result_ptr_c.ptr.add(i * vec_size),
                                                    vec_size
                                                );
                                            }
                                        }
                                        for i in inner..inner + remain {
                                            let a_val = a_data_ptr[i];
                                            let result_val = result_ptr_c[i];
                                            let mut_ref = unsafe {
                                                &mut *result_ptr_c.ptr.offset(i as isize)
                                            };
                                            *mut_ref = op(result_val, a_val);
                                        }
                                        next1(&mut iterator, &mut a_data_ptr);
                                    }
                                    next2(&mut iterator, &mut a_data_ptr);
                                    if let Some(op3) = op3 {
                                        for i in 0..inner_loop_size as i64 {
                                            let result_val = result_ptr_c[i];
                                            let mut_ref = unsafe {
                                                &mut *result_ptr_c.ptr.offset(i as isize)
                                            };
                                            *mut_ref = op3(result_val);
                                        }
                                    }
                                    result_ptr_c.add(inner_loop_size);
                                    iterator.reset_prg();
                                }
                            } else {
                            }

                            #[cfg(not(feature = "simd"))]
                            for _i in 0..current_size {
                                for _ in 0..inner_loop_size_2 {
                                    for i in 0..inner_loop_size as i64 {
                                        let a_val = a_data_ptr[i * a_last_stride];
                                        let result_val = result_ptr_c[i];
                                        let mut_ref = unsafe {
                                            &mut *result_ptr_c.ptr.offset(i as isize)
                                        };
                                        *mut_ref = op(result_val, a_val);
                                    }
                                    for j in (shape_len..=(iterator.a_shape.len() as i64) -
                                        1).rev() {
                                        if iterator.prg[j as usize] < iterator.a_shape[j as usize] {
                                            iterator.prg[j as usize] += 1;
                                            a_data_ptr.offset(iterator.strides[j as usize]);
                                            break;
                                        } else {
                                            iterator.prg[j as usize] = 0;
                                            a_data_ptr.offset(
                                                -iterator.strides[j as usize] *
                                                    iterator.a_shape[j as usize]
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
                                            -iterator.strides[j as usize] *
                                                iterator.a_shape[j as usize]
                                        );
                                    }
                                }
                                if let Some(op3) = op3 {
                                    for i in 0..inner_loop_size as i64 {
                                        let result_val = result_ptr_c[i];
                                        let mut_ref = unsafe {
                                            &mut *result_ptr_c.ptr.offset(i as isize)
                                        };
                                        *mut_ref = op3(result_val);
                                    }
                                }
                                result_ptr_c.add(inner_loop_size);
                                iterator.reset_prg();
                            }

                            barrier_clone.wait();
                        });
                    }
                    barrier.wait();
                }
            }
        });
    }
    if let Some(new_shape) = new_shape {
        let result = result.reshape(new_shape)?;
        Ok(result)
    } else {
        Ok(result)
    }
}
// #[cfg(not(feature = "simd"))]
// #[cfg_attr(feature = "track_caller", track_caller)]
// pub(crate) fn reduce<T, F>(
//     a: &_Tensor<T>,
//     op: F,
//     axes: &[usize],
//     init_val: T,
//     keepdims: bool,
//     init_out: bool,
//     c: Option<_Tensor<T>>
// )
//     -> anyhow::Result<_Tensor<T>>
//     where T: CommonBounds, F: Fn(T, T) -> T + Sync + Send + 'static + Copy
// {
//     _reduce::<_, _, _, fn(T) -> T, T>(a, op, op, None, &axes, init_val, keepdims, init_out, c)
// }

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn reduce<T, F, F2>(
    a: &_Tensor<T>,
    op: F,
    vec_op: F2,
    axes: &[usize],
    init_val: T,
    keepdims: bool,
    init_out: bool,
    c: Option<_Tensor<T>>
)
    -> anyhow::Result<_Tensor<T>>
    where
        T: CommonBounds + tensor_types::into_scalar::IntoScalar<T>,
        F: Fn(T, T) -> T + Sync + Send + 'static + Copy,
        F2: Fn(<T as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec +
            Sync +
            Send +
            'static +
            Copy,
        <T as TypeCommon>::Vec: Copy + IntoVec<<T as TypeCommon>::Vec>
{
    _reduce::<_, _, _, fn(T) -> T, _, fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec, T>(
        a,
        op,
        op,
        None,
        vec_op,
        None,
        &axes,
        init_val,
        keepdims,
        init_out,
        c
    )
}

// #[cfg(not(feature = "simd"))]
// #[cfg_attr(feature = "track_caller", track_caller)]
// pub(crate) fn reduce2<T, F, F2, O>(
//     a: &_Tensor<T>,
//     op: F,
//     op2: F2,
//     axes: &[usize],
//     init_val: O,
//     keepdims: bool,
//     init_out: bool,
//     c: Option<_Tensor<O>>
// )
//     -> anyhow::Result<_Tensor<O>>
//     where
//         T: CommonBounds,
//         F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
//         F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
//         O: CommonBounds
// {
//     _reduce::<_, _, _, fn(O) -> O, O>(a, op, op2, None, &axes, init_val, keepdims, init_out, c)
// }

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
    c: Option<_Tensor<O>>
)
    -> anyhow::Result<_Tensor<O>>
    where
        T: CommonBounds + IntoScalar<O>,
        F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
        F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
        F3: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec +
            Sync +
            Send +
            'static +
            Copy,
        O: CommonBounds,
        <T as TypeCommon>::Vec: IntoVec<<O as TypeCommon>::Vec> + Copy,
        <O as TypeCommon>::Vec: Copy
{
    _reduce::<T, F, F2, fn(O) -> O, _, fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec, O>(
        a,
        op,
        op2,
        None,
        vec_op,
        None,
        &axes,
        init_val,
        keepdims,
        init_out,
        c
    )
}

// #[cfg(not(feature = "simd"))]
// #[cfg_attr(feature = "track_caller", track_caller)]
// pub(crate) fn reduce3<T, F, F2, F3, O>(
//     a: &_Tensor<T>,
//     op: F,
//     op2: F2,
//     op3: F3,
//     axes: &[usize],
//     init_val: O,
//     keepdims: bool,
//     init_out: bool,
//     c: Option<_Tensor<O>>
// )
//     -> anyhow::Result<_Tensor<O>>
//     where
//         T: CommonBounds,
//         F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
//         F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
//         F3: Fn(O) -> O + Sync + Send + 'static + Copy,
//         O: CommonBounds
// {
//     _reduce::<_, _, _, _, O>(a, op, op2, Some(op3), &axes, init_val, keepdims, init_out, c)
// }

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
    c: Option<_Tensor<O>>
)
    -> anyhow::Result<_Tensor<O>>
    where
        T: CommonBounds + IntoScalar<O>,
        F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
        F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
        F3: Fn(O) -> O + Sync + Send + 'static + Copy,
        F4: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec +
            Sync +
            Send +
            'static +
            Copy,
        F5: Fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec + Sync + Send + 'static + Copy,
        O: CommonBounds,
        <T as TypeCommon>::Vec: IntoVec<<O as TypeCommon>::Vec> + Copy,
        <O as TypeCommon>::Vec: Copy
{
    _reduce::<T, F, F2, F3, F4, F5, O>(
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
        c
    )
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

// #[cfg(not(feature = "simd"))]
// impl<T: CommonBounds + NormalOut<Output = T> + Eval<Output = bool> + Cmp> NormalReduce<T>
// for _Tensor<T> {
//     type Output = _Tensor<T>;

//     type BoolOutput = _Tensor<bool>;

//     fn sum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce(self, |a, b| a._add(b), &axes, T::ZERO, keep_dims, false, None)
//     }

//     fn sum_<S: Into<Axis>>(
//         &self,
//         axes: S,
//         keep_dims: bool,
//         init_out: bool,
//         out: Self::Output
//     ) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce(self, |a, b| a._add(b), &axes, T::ZERO, keep_dims, init_out, Some(out))
//     }

//     fn sum_with_init<S: Into<Axis>>(
//         &self,
//         init_val: T,
//         axes: S,
//         keep_dims: bool
//     ) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce(self, |a, b| a._add(b), &axes, init_val, keep_dims, false, None)
//     }

//     fn nansum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce(
//             self,
//             |a, b| {
//                 if b._is_nan() { a } else { b._add(a) }
//             },
//             &axes,
//             T::ZERO,
//             keep_dims,
//             false,
//             None
//         )
//     }

//     fn nansum_with_init<S: Into<Axis>>(
//         &self,
//         init_val: T,
//         axes: S,
//         keep_dims: bool
//     ) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce(
//             self,
//             |a, b| {
//                 if b._is_nan() { a } else { b._add(a) }
//             },
//             &axes,
//             init_val,
//             keep_dims,
//             false,
//             None
//         )
//     }

//     fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axis, self.ndim())?;
//         reduce(self, |a, b| a._mul(b), &axes, T::ONE, keep_dims, false, None)
//     }

//     fn prod_with_init<S: Into<Axis>>(
//         &self,
//         init_val: T,
//         axes: S,
//         keep_dims: bool
//     ) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce(self, |a, b| a._mul(b), &axes, init_val, keep_dims, false, None)
//     }

//     fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axis, self.ndim())?;
//         reduce(
//             self,
//             |a, b| {
//                 if b._is_nan() { a } else { b._mul(a) }
//             },
//             &axes,
//             T::ONE,
//             keep_dims,
//             false,
//             None
//         )
//     }

//     fn nanprod_with_init<S: Into<Axis>>(
//         &self,
//         init_val: T,
//         axes: S,
//         keep_dims: bool
//     ) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce(
//             self,
//             |a, b| {
//                 if b._is_nan() { a } else { b._mul(a) }
//             },
//             &axes,
//             init_val,
//             keep_dims,
//             false,
//             None
//         )
//     }

//     fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
//         let axes: Vec<usize> = process_axes(axis, self.ndim())?;
//         reduce(self, |a, b| a._min(b), &axes, T::INF, keep_dims, false, None)
//     }

//     fn min_with_init<S: Into<Axis>>(
//         &self,
//         init_val: T,
//         axes: S,
//         keep_dims: bool
//     ) -> anyhow::Result<Self> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce(self, |a, b| a._min(b), &axes, init_val, keep_dims, false, None)
//     }

//     fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
//         let axes: Vec<usize> = process_axes(axis, self.ndim())?;
//         reduce(self, |a, b| a._max(b), &axes, T::NEG_INF, keep_dims, false, None)
//     }

//     fn max_with_init<S: Into<Axis>>(
//         &self,
//         init_val: T,
//         axes: S,
//         keep_dims: bool
//     ) -> anyhow::Result<Self> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce(self, |a, b| a._max(b), &axes, init_val, keep_dims, false, None)
//     }

//     fn all<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::BoolOutput> {
//         let axes: Vec<usize> = process_axes(axes, self.ndim())?;
//         reduce2(
//             self,
//             |a, b| b._is_true() & a,
//             |a, b| b._is_true() & a,
//             &axes,
//             true,
//             keep_dims,
//             false,
//             None
//         )
//     }

//     fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::BoolOutput> {
//         let axes: Vec<usize> = process_axes(axis, self.ndim())?;
//         reduce2(
//             self,
//             |a, b| b._is_true() | a,
//             |a, b| b._is_true() | a,
//             &axes,
//             false,
//             keep_dims,
//             false,
//             None
//         )
//     }

//     fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axis, self.ndim())?;
//         reduce(self, |a, b| a._add(b._abs()), &axes, T::ZERO, keep_dims, false, None)
//     }

//     fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
//         let axes: Vec<usize> = process_axes(axis, self.ndim())?;
//         reduce(self, |a, b| a._add(b._square()), &axes, T::ZERO, keep_dims, false, None)
//     }
// }

impl<T: CommonBounds + NormalOut<Output = T> + Eval<Output = bool> + Cmp> NormalReduce<T>
    for _Tensor<T>
    where
        <T as TypeCommon>::Vec: IntoVec<<T as TypeCommon>::Vec> +
            Copy +
            NormalOut<Output = <T as TypeCommon>::Vec> +
            Eval<Output = bool> +
            Cmp,
        T: IntoScalar<T> + IntoScalar<bool>
{
    type Output = _Tensor<T>;

    type BoolOutput = _Tensor<bool>;

    fn sum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None
        )
    }

    fn sum_<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        out: Self::Output
    ) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            Some(out)
        )
    }

    fn sum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }

    fn nansum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| {
                if b._is_nan() { a } else { b._add(a) }
            },
            |a, b| {
                if b._is_nan() { a } else { b._add(a) }
            },
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None
        )
    }

    fn nansum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| {
                if b._is_nan() { a } else { b._add(a) }
            },
            |a, b| {
                if b._is_nan() { a } else { b._add(a) }
            },
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }

    fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            &axes,
            T::ONE,
            keep_dims,
            false,
            None
        )
    }

    fn prod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }

    fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| {
                if b._is_nan() { a } else { b._mul(a) }
            },
            |a, b| {
                if b._is_nan() { a } else { b._mul(a) }
            },
            &axes,
            T::ONE,
            keep_dims,
            false,
            None
        )
    }

    fn nanprod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| {
                if b._is_nan() { a } else { b._mul(a) }
            },
            |a, b| {
                if b._is_nan() { a } else { b._mul(a) }
            },
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }

    fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._min(b),
            |a, b| a._min(b),
            &axes,
            T::INF,
            keep_dims,
            false,
            None
        )
    }

    fn min_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._min(b),
            |a, b| a._min(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }

    fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._max(b),
            |a, b| a._max(b),
            &axes,
            T::NEG_INF,
            keep_dims,
            false,
            None
        )
    }

    fn max_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._max(b),
            |a, b| a._max(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }

    fn all<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::BoolOutput> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        // reduce2(
        //     self,
        //     |a, b| b & a,
        //     |a, b| b & a,
        //     |a, b| {let mut res = [false; 32];
        //         for i in 0..32 {
        //             res[i] = b[i] & a[i];
        //         }
        //         res
        //     },
        //     &axes,
        //     true,
        //     keep_dims,
        //     false,
        //     None
        // )
        todo!()
    }

    fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::BoolOutput> {
        // let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        // reduce2(
        //     self,
        //     |a, b| b | a,
        //     |a, b| b | a,
        //     |a, b| {
        //         let mut res = [false; 32];
        //         for i in 0..32 {
        //             res[i] = b[i] | a[i];
        //         }
        //         res
        //     },
        //     &axes,
        //     false,
        //     keep_dims,
        //     false,
        //     None
        // )
        todo!()
    }

    fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b._abs()),
            |a, b| a._add(b._abs()),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None
        )
    }

    fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b._square()),
            |a, b| a._add(b._square()),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None
        )
    }
}

impl<T> FloatReduce<T>
    for _Tensor<T>
    where
        T: CommonBounds                                                                                 // prettier-ignore
        + NormalOut<T, Output = T>                                                                                  // prettier-ignore
        + NormalOut<FloatType<T>, Output = FloatType<T>>                          // prettier-ignore
        + FloatOut + Cmp + IntoScalar<T>
        + IntoScalar<<T as FloatOut>::Output>, // prettier-ignore
        FloatType<T>: CommonBounds                                                           // prettier-ignore
        + NormalOut<T, Output = FloatType<T>>
        + FloatOut<Output = FloatType<T>>
        + NormalOut<FloatType<T>, Output = FloatType<T>> // prettier-ignore
        + FromScalar<usize> + IntoScalar<FloatType<T>>, // prettier-ignore
        f64: IntoScalar<<T as NormalOut>::Output>, // prettier-ignore
        f64: IntoScalar<FloatType<T>>, // prettier-ignore
        _Tensor<<T as FloatOut>::Output>: TensorLike<
            <T as FloatOut>::Output,
            Output = _Tensor<<T as FloatOut>::Output>
        >,
        <<T as FloatOut>::Output as TypeCommon>::Vec: 
        NormalOut<Output = <FloatType<T> as TypeCommon>::Vec>
        + FloatOut<Output = <FloatType<T> as TypeCommon>::Vec>
        + IntoVec<<FloatType<T> as TypeCommon>::Vec> + Copy + Send + Sync, // prettier-ignore
        <T as TypeCommon>::Vec: IntoVec<<FloatType<T> as TypeCommon>::Vec>,
        <T as TypeCommon>::Vec: std::marker::Copy
{
    type Output = _Tensor<FloatType<T>>;
    fn mean<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        // let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        // let reduce_size: FloatType<T> = (
        //     axes.iter().fold(1, |acc, &x| acc * (self.shape()[x] as usize)) as f64
        // ).into_scalar();
        // let reduce_vec = <FloatType<T> as TypeCommon>::Vec::splat(reduce_size);
        // reduce3(
        //     self,
        //     |a, b| a._add(b),
        //     |a, b| a._add(b),
        //     move |a| a._div(reduce_size),
        //     |a, b| a._add(b),
        //     move |a| a._div(reduce_vec),
        //     &axes,
        //     <T as FloatOut>::Output::ZERO,
        //     keep_dims,
        //     false,
        //     None
        // )
        todo!()
    }

    fn reducel2<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        // let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        // reduce3(
        //     self,
        //     |a: <T as FloatOut>::Output, b| {
        //         let b = <<T as FloatOut>::Output as NormalOut>::_square(b);
        //         a._add(b)
        //     },
        //     |a, b| a._add(b),
        //     move |a| a._sqrt(),
        //     |a, b| a._add(b),
        //     |a| a._sqrt(),
        //     &axes,
        //     <T as FloatOut>::Output::ZERO,
        //     keep_dims,
        //     false,
        //     None
        // )
        todo!()
    }

    fn reducel3<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        // let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        // let three: <T as FloatOut>::Output = (3.0).into_scalar();
        // let three_vec = <<T as FloatOut>::Output as TypeCommon>::Vec::splat(three);
        // reduce3(
        //     self,
        //     move |a, b| {
        //         let b = <<T as FloatOut>::Output as NormalOut>::_abs(b);
        //         a._add(b._pow(three))
        //     },
        //     move |a, b| a._add(<FloatType<T> as NormalOut>::_abs(b)._pow(three)),
        //     move |a| a,
        //     move |a, b|
        //         a._add(<<<T as FloatOut>::Output as TypeCommon>::Vec>::_abs(b)._pow(three_vec)),
        //     |a| a,
        //     &axes,
        //     <T as FloatOut>::Output::ZERO,
        //     keep_dims,
        //     false,
        //     None
        // )
        todo!()
    }

    fn logsumexp<S: Into<Axis>>(&self, _: S, _: bool) -> anyhow::Result<Self::Output> {
        // let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        // let x_max = reduce(self, |a, b| a._max(b), &axes, T::NEG_INF, true, false, None)?;
        // let sub = self - &x_max;
        // let exp = sub.exp()?;
        // let sum_exp = reduce(
        //     &exp,
        //     |a, b| a._add(b),
        //     &axes,
        //     <T as FloatOut>::Output::ZERO,
        //     true,
        //     false,
        //     None
        // )?;
        // let add = x_max + sum_exp.ln()?;
        // if keep_dims {
        //     Ok(add)
        // } else {
        //     Ok(add.squeeze(axes)?)
        // }
        todo!()
    }
}
