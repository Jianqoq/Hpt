use std::borrow::BorrowMut;

use hpt_common::{
    error::{base::TensorError, shape::ShapeError},
    shape::shape::Shape,
    shape::shape_utils::mt_intervals,
    strides::strides::Strides,
    utils::pointer::Pointer,
};
use hpt_traits::{CommonBounds, ShapeManipulate, TensorCreator, TensorInfo};
use hpt_types::into_scalar::Cast;

use crate::{
    ops::common::reduce::{is_keep_fast_dim, rearrange_array},
    tensor_base::_Tensor,
    Cpu,
};

#[derive(Debug, Clone)]
pub(crate) struct NormalizePreprocessor<T, U> {
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

impl<T, U> NormalizePreprocessor<T, U>
where
    T: Clone,
    U: Clone,
{
    pub fn new(
        num_threads: usize,
        loop_size: usize,
        ptrs: Pointer<T>,
        res_ptrs: Pointer<U>,
        strides: Strides,
        res_strides: Strides,
        a_shape: Shape,
        transposed_shape: Shape,
        reduce_shape: Shape,
    ) -> Vec<NormalizePreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators: Vec<NormalizePreprocessor<T, U>> = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; reduce_shape.len()];
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let mut res_ptrs_cpy = res_ptrs.clone();
            let a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();
            let res_ptrs_cpy = res_ptrs_cpy.borrow_mut();
            /*traverse the whole result shape and increment the input data ptr based on current thread id*/
            for i in (0..=reduce_shape.len() - 1).rev() {
                a_data_ptr_cpy.offset(progress_init_a_data[i] * strides[i]);
                res_ptrs_cpy.offset(progress_init_a_data[i] * res_strides[i]);
            }
            // calculate the total task amount so far based on current thread id,
            // we are splitting the whole tensor into two axes
            // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]               thread 0
            // [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]     thread 1
            // [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]     thread 2
            // [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]     thread 3
            // [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]     thread 4
            // where the first axis is where we are splitting the tensor
            let mut tmp1 = task_amout as i64;
            let mut prg = vec![0; a_shape.len() - 1]; /* -1 because we want to escape the last axis */

            // since the axis we want to reduce include the most inner axis, we will skip the iteration of the last axis
            // so we use (0..=a_shape.len() - 2).rev()
            for i in (0..=a_shape.len() - 2).rev() {
                prg[i] = tmp1 % transposed_shape[i];
                tmp1 /= transposed_shape[i];
            }

            // increment the res ptr based on the current thread task amount for next thread (next iteration)
            task_amout += intervals[id].1 - intervals[id].0;

            let mut tmp2 = task_amout as i64;
            for j in (0..=reduce_shape.len() - 1).rev() {
                progress_init_a_data[j] = tmp2 % reduce_shape[j];
                tmp2 /= reduce_shape[j];
            }
            iterators.push(NormalizePreprocessor {
                ptrs: a_data_ptr_cpy.clone(),
                res_ptrs: res_ptrs_cpy.clone(),
                strides: strides.clone(),
                start: intervals[id].0,
                end: intervals[id].1,
                prg,
                a_prg: vec![],
                shape: reduce_shape.clone(),
                a_shape: a_shape.clone(),
            });
        }
        iterators
    }

    pub fn new2(
        num_threads: usize,
        loop_size: usize,
        ptrs: Pointer<T>,
        res_ptrs: Pointer<U>,
        transposed_strides: Strides,
        res_transposed_strides: Strides,
        transposed_shape: Shape,
        reduce_shape: Shape,
    ) -> Vec<NormalizePreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; reduce_shape.len()];
        let ndim = reduce_shape.len() as i64;

        // [0, 6, 12, 18, 24, 30] res0    thread 0
        // [1, 7, 13, 19, 25, 31] res0    thread 0
        // [2, 8, 14, 20, 26, 32] res1    thread 1
        // [3, 9, 15, 21, 27, 33] res1    thread 1
        // [4, 10, 16, 22, 28, 34] res2   thread 2
        // [5, 11, 17, 23, 29, 35] res2   thread 2
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let mut res_ptr_cpy = res_ptrs.clone();
            let a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();
            let res_ptr_cpy = res_ptr_cpy.borrow_mut();

            for i in (0..ndim - 1).rev() {
                a_data_ptr_cpy
                    .offset(progress_init_a_data[i as usize] * transposed_strides[i as usize]);
                res_ptr_cpy
                    .offset(progress_init_a_data[i as usize] * res_transposed_strides[i as usize]);
            }

            let progress_init_a_data_cpy = progress_init_a_data.clone();

            task_amout += intervals[id].1 - intervals[id].0;

            let prg = vec![0; transposed_shape.len()];

            let mut tmp = task_amout as i64;
            for j in (0..ndim - 1).rev() {
                progress_init_a_data[j as usize] = tmp % reduce_shape[j as usize];
                tmp /= reduce_shape[j as usize];
            }

            iterators.push(NormalizePreprocessor {
                ptrs: a_data_ptr_cpy.clone(),
                res_ptrs: res_ptr_cpy.clone(),
                strides: transposed_strides.clone(),
                start: intervals[id].0,
                end: intervals[id].1,
                prg,
                a_prg: progress_init_a_data_cpy,
                shape: reduce_shape.clone(),
                a_shape: transposed_shape.clone(),
            });
        }
        iterators
    }
}

#[derive(Debug, Clone)]
pub(crate) struct UCNormalizePreprocessor<T, U> {
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

impl<T, U> UCNormalizePreprocessor<T, U>
where
    T: Clone,
    U: Clone,
{
    pub fn new2(
        num_threads: usize,
        loop_size: usize,
        ptrs: Pointer<T>,
        res_ptrs: Pointer<U>,
        transposed_strides: Strides,
        transposed_shape: Shape,
        reduce_shape: Shape,
        res_strides: Strides,
    ) -> Vec<UCNormalizePreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; reduce_shape.len()];
        let ndim = reduce_shape.len() as i64;
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();
            let mut res_ptrs_cpy = res_ptrs.clone();
            let res_ptrs_cpy = res_ptrs_cpy.borrow_mut();

            for i in (0..ndim - 1).rev() {
                a_data_ptr_cpy
                    .offset(progress_init_a_data[i as usize] * transposed_strides[i as usize]);
                res_ptrs_cpy.offset(progress_init_a_data[i as usize] * res_strides[i as usize]);
            }

            let progress_init_a_data_cpy = progress_init_a_data.clone();

            task_amout += intervals[id].1 - intervals[id].0;

            let prg = vec![0; transposed_shape.len()];

            let mut tmp = task_amout as i64;
            for j in (0..ndim - 1).rev() {
                progress_init_a_data[j as usize] = tmp % reduce_shape[j as usize];
                tmp /= reduce_shape[j as usize];
            }

            iterators.push(UCNormalizePreprocessor {
                ptrs: a_data_ptr_cpy.clone(),
                res_ptrs: res_ptrs_cpy.clone(),
                strides: transposed_strides.clone(),
                start: intervals[id].0,
                end: intervals[id].1,
                prg,
                a_prg: progress_init_a_data_cpy,
                shape: reduce_shape.clone(),
                a_shape: transposed_shape.clone(),
            });
        }
        iterators
    }
}

pub(crate) fn normalize_prepare<T: CommonBounds, O: CommonBounds, const DEVICE: usize>(
    a: &_Tensor<T, Cpu, DEVICE>,
    axis: usize,
    c: Option<_Tensor<O, Cpu, DEVICE>>,
) -> std::result::Result<(bool, _Tensor<T, Cpu, DEVICE>, _Tensor<O, Cpu, DEVICE>), TensorError> {
    let keep_fast_dim = is_keep_fast_dim(&a.layout.strides(), &[axis]);
    // get permute order, we move to_reduce axes to the end
    let mut transposed_axis = rearrange_array(a.ndim(), &[axis]);

    // sort the transposed axis based on the stride, ordering the axis can increase the cpu cache hitting rate when we do iteration
    transposed_axis[..a.ndim() - 1].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));
    transposed_axis[a.ndim() - 1..].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));

    let res = if let Some(out) = c {
        // we need a better logic to verify the out is valid.
        // we need to get the real size and compare the real size with the res_shape
        ShapeError::check_inplace_out_layout_valid(a.shape(), out.layout())?;
        Ok(out)
    } else {
        _Tensor::<O, Cpu, DEVICE>::empty(a.shape())
    };
    Ok((keep_fast_dim, a.permute(transposed_axis)?, res?))
}

#[track_caller]
pub(crate) fn contiguous_normalize_template<T, F1, F2, F3, O, const DEVICE: usize>(
    a: &_Tensor<T, Cpu, DEVICE>,
    axis: usize,
    c: Option<_Tensor<O, Cpu, DEVICE>>,
    full_reduce: F1,
    nkd: F2,
    kd: F3,
) -> Result<_Tensor<O, Cpu, DEVICE>, TensorError>
where
    T: CommonBounds + Cast<O>,
    O: CommonBounds,
    F1: Fn(&mut O),
    F2: Fn(usize, usize, &_Tensor<O, Cpu, DEVICE>, &_Tensor<T, Cpu, DEVICE>),
    F3: Fn(usize, usize, usize, &_Tensor<O, Cpu, DEVICE>, &_Tensor<T, Cpu, DEVICE>),
{
    let (keep_fast_dim, transposed_tensor, result) = normalize_prepare(a, axis, c)?;

    let a_last_stride = if keep_fast_dim {
        transposed_tensor.strides()[a.ndim() - 2]
    } else {
        transposed_tensor.strides()[a.ndim() - 1]
    };
    let inner_loop_size = if keep_fast_dim {
        transposed_tensor.shape()[a.ndim() - 2]
    } else {
        transposed_tensor.shape()[a.ndim() - 1]
    } as usize;
    assert_eq!(a_last_stride, 1);
    let result_data = result.ptr();
    if a.ndim() == 1 {
        full_reduce(unsafe { result_data.get_ptr().as_mut().unwrap() });
    } else {
        if !keep_fast_dim {
            let num_threads = if result.size() < rayon::current_num_threads() {
                result.size()
            } else {
                rayon::current_num_threads()
            };
            nkd(num_threads, inner_loop_size, &result, &transposed_tensor);
        } else {
            let a_reduce_size = a.size() / (a.shape()[axis] as usize);
            let outer_loop_size = a_reduce_size / inner_loop_size;
            let inner_loop_size_2 = a.shape()[axis] as usize;
            let num_threads = if outer_loop_size < rayon::current_num_threads() {
                outer_loop_size
            } else {
                rayon::current_num_threads()
            };
            kd(
                num_threads,
                inner_loop_size,
                inner_loop_size_2,
                &result,
                &transposed_tensor,
            );
        }
    }
    Ok(result)
}

#[track_caller]
pub(crate) fn uncontiguous_normalize_template<T, F1, F2, F3, O, const DEVICE: usize>(
    a: &_Tensor<T, Cpu, DEVICE>,
    axis: usize,
    c: Option<_Tensor<O, Cpu, DEVICE>>,
    full_reduce: F1,
    nkd: F2,
    kd: F3,
) -> std::result::Result<_Tensor<O, Cpu, DEVICE>, TensorError>
where
    T: CommonBounds + Cast<O>,
    O: CommonBounds,
    F1: Fn(&mut O),
    F2: Fn(usize, usize, &_Tensor<O, Cpu, DEVICE>, &_Tensor<T, Cpu, DEVICE>),
    F3: Fn(usize, usize, usize, &_Tensor<O, Cpu, DEVICE>, &_Tensor<T, Cpu, DEVICE>),
{
    let (keep_fast_dim, transposed_tensor, result) = normalize_prepare(a, axis, c)?;

    let result_data = result.ptr();
    if a.ndim() == 1 {
        full_reduce(unsafe { result_data.get_ptr().as_mut().unwrap() });
    } else {
        let inner_loop_size = (if keep_fast_dim {
            transposed_tensor.shape()[a.ndim() - 2]
        } else {
            transposed_tensor.shape()[a.ndim() - 1]
        }) as usize;
        if !keep_fast_dim {
            let num_threads = if result.size() < rayon::current_num_threads() {
                result.size()
            } else {
                rayon::current_num_threads()
            };
            nkd(num_threads, inner_loop_size, &result, &transposed_tensor);
        } else {
            let a_reduce_size = a.size() / (a.shape()[axis] as usize);
            let outer_loop_size = a_reduce_size / inner_loop_size;
            let inner_loop_size_2 = a.shape()[axis] as usize;
            let num_threads = if outer_loop_size < rayon::current_num_threads() {
                outer_loop_size
            } else {
                rayon::current_num_threads()
            };
            assert!(inner_loop_size > 1);
            kd(
                num_threads,
                inner_loop_size,
                inner_loop_size_2,
                &result,
                &transposed_tensor,
            );
        }
    }
    Ok(result)
}
