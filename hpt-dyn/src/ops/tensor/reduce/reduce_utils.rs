use std::borrow::BorrowMut;

use hpt_common::{error::{base::TensorError, shape::ShapeError}, shape::{shape::Shape, shape_utils::mt_intervals}, strides::strides::Strides, Pointer};
use hpt_types::dtype::DType;

use crate::Tensor;

pub(crate) fn rearrange_array(ndim: usize, to_reduce: &[usize]) -> Vec<usize> {
    let mut origin_order = (0..ndim).collect::<Vec<usize>>();
    let mut to_reduce = to_reduce.to_vec();
    // sort the reduce axes
    to_reduce.sort();

    // store the elements to be reduced
    let mut moved_elements = Vec::new();
    origin_order.retain(|&x| {
        if to_reduce.contains(&x) {
            moved_elements.push(x);
            false
        } else {
            true
        }
    });

    // put the reduced elements at the end
    origin_order.extend(moved_elements);

    origin_order
}

pub(crate) fn is_keep_fast_dim(strides: &[i64], axes: &[usize]) -> bool {
    let mut keep_fast_dim = true;
    for axis in axes.iter() {
        if strides[*axis] == 1 {
            keep_fast_dim = false;
            break;
        }
    }
    keep_fast_dim
}

pub(crate) fn get_fast_dim_size(shape: &[i64], strides: &[i64], axes: &[usize]) -> i64 {
    for axis in axes.iter() {
        if strides[*axis] == 1 {
            return shape[*axis];
        }
    }
    unreachable!()
}

pub(crate) fn split_groups_by_axes(groups: &Vec<Vec<usize>>, axes: &[usize]) -> Vec<Vec<usize>> {
    let mut result = Vec::new();

    for group in groups {
        let mut current_group = Vec::new();
        let mut current_is_reduce = false;
        let mut first_dim = true;

        for &dim in group {
            let is_reduce = axes.contains(&dim);

            if first_dim {
                current_is_reduce = is_reduce;
                current_group.push(dim);
                first_dim = false;
            } else if is_reduce == current_is_reduce {
                current_group.push(dim);
            } else {
                result.push(current_group);
                current_group = vec![dim];
                current_is_reduce = is_reduce;
            }
        }

        if !current_group.is_empty() {
            result.push(current_group);
        }
    }

    result
}

pub(crate) fn get_new_reduce_axes(groups: Vec<Vec<usize>>, axes: &[usize]) -> Vec<usize> {
    let mut result = vec![];
    for (idx, group) in groups.into_iter().enumerate() {
        if group.iter().any(|&x| axes.contains(&x)) {
            let all_reduce = group.iter().all(|&x| axes.contains(&x));
            if !all_reduce {
                panic!("Inconsistent reduction properties in dimension group");
            }
            result.push(idx);
        }
    }
    result
}

pub(crate) fn get_new_shape(groups: &Vec<Vec<usize>>, shape: &[i64]) -> Vec<i64> {
    let mut result = vec![];
    for group in groups {
        result.push(group.iter().map(|&x| shape[x]).product::<i64>());
    }
    result
}

pub(crate) fn reduce_prepare(
    a: &Tensor,
    axes: &[usize],
    init_val: f64,
    init_out: bool,
    res_dtype: DType,
    c: Option<Tensor>,
) -> std::result::Result<(Tensor, Tensor), TensorError> {
    // get permute order, we move to_reduce axes to the end
    let mut transposed_axis = rearrange_array(a.ndim(), axes);

    // sort the transposed axis based on the stride, ordering the axis can increase the cpu cache hitting rate when we do iteration
    transposed_axis[..a.ndim() - axes.len()].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));
    transposed_axis[a.ndim() - axes.len()..].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));
    let res_layout = a.layout.reduce(axes, false)?;

    let res = if let Some(mut out) = c {
        // we need a better logic to verify the out is valid.
        // we need to get the real size and compare the real size with the res_shape
        ShapeError::check_inplace_out_layout_valid(res_layout.shape(), &out.layout)?;
        assert_eq!(out.dtype, res_dtype);
        if init_out {
            out.fill(init_val);
        }
        Ok(out.reshape(res_layout.shape())?)
    } else {
        Tensor::full(res_layout.shape(), init_val, res_dtype, a.device.clone())
    };
    let transposed_axis = transposed_axis
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();
    Ok((a.permute(&transposed_axis)?, res?))
}

pub(crate) fn uncontiguous_reduce_prepare(
    a: &Tensor,
    axes: &[usize],
    init_val: f64,
    init_out: bool,
    res_dtype: DType,
    c: Option<Tensor>,
) -> std::result::Result<(bool, Tensor, Tensor, Vec<i64>), TensorError> {
    let keep_fast_dim = is_keep_fast_dim(&a.layout.strides(), axes);
    // get permute order, we move to_reduce axes to the end
    let mut transposed_axis = rearrange_array(a.ndim(), axes);
    // sort the transposed axis based on the stride, ordering the axis can increase the cpu cache hitting rate when we do iteration
    transposed_axis[..a.ndim() - axes.len()].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));
    transposed_axis[a.ndim() - axes.len()..].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));
    let res_layout = a.layout.reduce(axes, false)?;

    let mut res_permute_axes = (0..res_layout.ndim()).collect::<Vec<usize>>();
    res_permute_axes.sort_by(|a, b| transposed_axis[*a].cmp(&transposed_axis[*b]));

    let res_permute_axes = res_permute_axes
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();
    let transposed_axis = transposed_axis
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();
    let res = if let Some(mut out) = c {
        // we need a better logic to verify the out is valid.
        // we need to get the real size and compare the real size with the res_shape
        ShapeError::check_inplace_out_layout_valid(res_layout.shape(), &out.layout)?;
        assert_eq!(out.dtype, res_dtype);
        if init_out {
            out.fill(init_val);
        }
        Ok(out)
    } else {
        Tensor::full(res_layout.shape(), init_val, res_dtype, a.device.clone())?
            .permute(&res_permute_axes)
    };
    Ok((
        keep_fast_dim,
        a.permute(&transposed_axis)?,
        res?,
        res_permute_axes,
    ))
}


#[derive(Debug, Clone)]
pub(crate) struct UCReductionPreprocessor<T, U> {
    pub ptrs: Pointer<T>,
    pub res_ptrs: Pointer<U>,
    pub strides: Strides,
    pub start: usize,
    pub end: usize,
    pub prg: Vec<i64>,
    pub a_prg: Vec<i64>,
    pub shape: Shape,
    pub a_shape: Shape,
    pub res_prg: Vec<i64>,
}

impl<T, U> UCReductionPreprocessor<T, U>
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
        res_strides: &[i64],
    ) -> Vec<UCReductionPreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators: Vec<UCReductionPreprocessor<T, U>> = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; res_shape.len()];
        let res_ptrs = res_ptrs.borrow_mut();
        for id in 0..num_threads {
            let mut res_prg = vec![0; res_shape.len()];
            let mut a_data_ptr_cpy = ptrs.clone();
            let mut a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();

            /*traverse the whole result shape and increment the input data ptr based on current thread id*/
            for i in (0..=res_shape.len() - 1).rev() {
                a_data_ptr_cpy += progress_init_a_data[i] * strides[i];
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
            for i in (0..a_shape.len() - 1).rev() {
                prg[i] = tmp1 % transposed_shape[i];
                tmp1 /= transposed_shape[i];
            }
            // increment the res ptr based on the current thread task amount for next thread (next iteration)
            task_amout += intervals[id].1 - intervals[id].0;
            let mut res_ptr_cpy = res_ptrs.clone();
            let mut tmp = intervals[id].0 as i64;
            for i in (0..res_shape.len()).rev() {
                res_prg[i] = tmp % res_shape[i];
                tmp /= res_shape[i];
                res_ptr_cpy += res_prg[i] * res_strides[i];
            }

            let mut tmp2 = task_amout as i64;
            for j in (0..=res_shape.len() - 1).rev() {
                progress_init_a_data[j] = tmp2 % res_shape[j];
                tmp2 /= res_shape[j];
            }
            iterators.push(UCReductionPreprocessor {
                ptrs: a_data_ptr_cpy.clone(),
                res_ptrs: res_ptr_cpy,
                strides: strides.clone(),
                start: intervals[id].0,
                end: intervals[id].1,
                prg,
                a_prg: vec![],
                shape: res_shape.clone(),
                a_shape: a_shape.clone(),
                res_prg,
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
        res_strides: &[i64],
    ) -> Vec<UCReductionPreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; res_shape.len()];
        let res_ptrs = res_ptrs.borrow_mut();
        let ndim = res_shape.len() as i64;
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let mut a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();

            for i in (0..ndim - 1).rev() {
                a_data_ptr_cpy += progress_init_a_data[i as usize] * transposed_strides[i as usize];
            }

            let progress_init_a_data_cpy = progress_init_a_data.clone();

            task_amout += intervals[id].1 - intervals[id].0;

            let prg = vec![0; transposed_shape.len()];
            let mut res_prg = vec![0; res_shape.len()];
            let mut res_ptr_cpy = res_ptrs.clone();
            let mut res_task_amout = (intervals[id].0 * inner_loop_size) as i64;
            for j in (0..res_shape.len()).rev() {
                res_prg[j] = res_task_amout % res_shape[j];
                res_task_amout /= res_shape[j];
                res_ptr_cpy += res_prg[j] * res_strides[j];
            }

            let mut tmp = task_amout as i64;
            for j in (0..ndim - 1).rev() {
                progress_init_a_data[j as usize] = tmp % res_shape[j as usize];
                tmp /= res_shape[j as usize];
            }

            iterators.push(UCReductionPreprocessor {
                ptrs: a_data_ptr_cpy.clone(),
                res_ptrs: res_ptr_cpy,
                strides: transposed_strides.clone(),
                start: intervals[id].0,
                end: intervals[id].1,
                prg,
                a_prg: progress_init_a_data_cpy,
                shape: res_shape.clone(),
                a_shape: transposed_shape.clone(),
                res_prg,
            });
        }
        iterators
    }
}

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
            let mut a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();

            /*traverse the whole result shape and increment the input data ptr based on current thread id*/
            for i in (0..=res_shape.len() as i64 - 1).rev() {
                a_data_ptr_cpy += progress_init_a_data[i as usize] * strides[i as usize];
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
            for i in (0..=a_shape.len() as i64 - 2).rev() {
                prg[i as usize] = tmp1 % transposed_shape[i as usize];
                tmp1 /= transposed_shape[i as usize];
            }

            // increment the res ptr based on the current thread task amount for next thread (next iteration)
            task_amout += intervals[id].1 - intervals[id].0;
            let res_ptr_cpy = res_ptrs.clone();
            res_ptrs.add(intervals[id].1 - intervals[id].0);

            let mut tmp2 = task_amout as i64;
            for j in (0..=res_shape.len() as i64 - 1).rev() {
                progress_init_a_data[j as usize] = tmp2 % res_shape[j as usize];
                tmp2 /= res_shape[j as usize];
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
        // [1, 7, 13, 19, 25, 31] res1    thread 0
        // [2, 8, 14, 20, 26, 32] res2    thread 1
        // [3, 9, 15, 21, 27, 33] res3    thread 1
        // [4, 10, 16, 22, 28, 34] res4   thread 2
        // [5, 11, 17, 23, 29, 35] res5   thread 2
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let mut a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();

            for i in (0..ndim - 1).rev() {
                a_data_ptr_cpy += progress_init_a_data[i as usize] * transposed_strides[i as usize];
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
