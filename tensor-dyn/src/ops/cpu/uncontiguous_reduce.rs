use std::borrow::BorrowMut;
use std::sync::Arc;
use std::sync::Barrier;
use tensor_common::shape_utils::mt_intervals;
use tensor_types::traits::VecCommon;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use tensor_common::pointer::Pointer;
use tensor_common::slice::Slice;
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_types::{ dtype::TypeCommon, into_scalar::IntoScalar };
use tensor_common::shape_utils::predict_reduce_shape;
use crate::backend::Cpu;
use crate::slice::SliceOps;
use crate::tensor_base::_Tensor;
use crate::THREAD_POOL;
use tensor_traits::TensorInfo;
use tensor_traits::ShapeManipulate;

#[derive(Debug)]
pub(crate) struct ReductionPreprocessor<T, U> {
    pub ptrs: Pointer<T>,
    pub res_ptrs: Pointer<U>,
    pub strides: Vec<i64>,
    pub start: usize,
    pub end: usize,
    pub prg: Vec<i64>,
    pub a_prg: Vec<i64>,
    pub shape: Arc<Vec<i64>>,
    pub a_shape: Arc<Vec<i64>>,
    pub res_prg: Vec<i64>,
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
        res_shape: Arc<Vec<i64>>,
        res_strides: &[i64]
    ) -> Vec<ReductionPreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators: Vec<ReductionPreprocessor<T, U>> = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; res_shape.len()];
        let res_ptrs = res_ptrs.borrow_mut();
        for id in 0..num_threads {
            let mut res_prg = vec![0; res_shape.len()];
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
            for i in (0..a_shape.len() - 1).rev() {
                prg[i] = tmp1 % transposed_shape[i];
                tmp1 /= transposed_shape[i];
            }
            // increment the res ptr based on the current thread task amount for next thread (next iteration)
            task_amout += intervals[id].1 - intervals[id].0;
            let mut res_ptr_cpy = res_ptrs.clone();
            let mut tmp = intervals[id].0 as i64;
            let mut offset = 0;
            for i in (0..res_shape.len()).rev() {
                res_prg[i] = tmp % res_shape[i];
                tmp /= res_shape[i];
                offset += res_prg[i] * res_strides[i];
            }
            res_ptr_cpy.offset(offset);

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
        transposed_strides: Vec<i64>,
        transposed_shape: Arc<Vec<i64>>,
        res_shape: Arc<Vec<i64>>,
        res_strides: &[i64]
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

            for i in (0..ndim - 1).rev() {
                a_data_ptr_cpy.offset(
                    progress_init_a_data[i as usize] * transposed_strides[i as usize]
                );
            }

            let progress_init_a_data_cpy = progress_init_a_data.clone();

            task_amout += intervals[id].1 - intervals[id].0;

            let prg = vec![0; transposed_shape.len()];
            let mut res_prg = vec![0; res_shape.len()];
            let mut res_ptr_cpy = res_ptrs.clone();
            let mut res_task_amout = (intervals[id].0 * inner_loop_size) as i64;
            let mut index = 0;
            for j in (0..res_shape.len()).rev() {
                res_prg[j] = res_task_amout % res_shape[j];
                res_task_amout /= res_shape[j];
                index += res_prg[j] * res_strides[j];
            }
            res_ptr_cpy.offset(index);

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
                res_prg,
            });
        }
        iterators
    }
}

fn rearrange_array(ndim: usize, to_reduce: &[usize]) -> Vec<usize> {
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

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn _reduce<T, F, F2, F3, F4, F5, O>(
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
    c: Option<_Tensor<O>>
)
    -> anyhow::Result<_Tensor<O>>
    where
        T: CommonBounds + IntoScalar<O> + tensor_types::convertion::Convertor,
        O: CommonBounds,
        F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
        F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
        F3: Fn(O) -> O + Sync + Send + 'static + Copy,
        F4: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec +
            'static +
            Copy +
            Send,
        F5: Fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec + Sync + Send + 'static + Copy,
        <T as TypeCommon>::Vec: Copy,
        <O as TypeCommon>::Vec: Copy
{
    use tensor_common::shape_utils::mt_intervals_simd;
    use tensor_iterator::iterator_traits::StridedIterator;

    let mut is_left = true;
    for axis in axes.iter() {
        if a.strides()[*axis] == 1 {
            is_left = false;
            break;
        }
    }
    let a_: &_Tensor<T> = &a;
    let a_shape = a_.shape();
    let a_shape_tmp = a_shape.clone();
    let (a_shape_cpy, res_shape) = predict_reduce_shape(&a_shape_tmp, &axes);
    let mut transposed_axis = rearrange_array(a_.ndim(), axes);
    transposed_axis[a.ndim() - axes.len()..].sort_by(|a, b| {
        a_.strides()[*b].cmp(&a_.strides()[*a])
    });
    transposed_axis[..a.ndim() - axes.len()].sort_by(|a, b| {
        a_.strides()[*b].cmp(&a_.strides()[*a])
    });

    // create permute axes for result also
    let mut res_permute_axes = (0..res_shape.len()).collect::<Vec<usize>>();
    res_permute_axes.sort_by(|a, b| { transposed_axis[*a].cmp(&transposed_axis[*b]) });

    let transposed_tensor = a_.permute(transposed_axis)?;
    let a_last_stride = if is_left {
        transposed_tensor.strides()[a.ndim() - axes.len() - 1]
    } else {
        transposed_tensor.strides()[a_.ndim() - 1]
    };
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
                return Err(anyhow::Error::msg("Output array has incorrect shape".to_string()));
            }
        } else {
            if res_shape.as_ref() != out.shape().inner() {
                return Err(anyhow::Error::msg("Output array has incorrect shape".to_string()));
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
        result = _Tensor::<O, Cpu>::full(init_val, res_shape.clone())?.permute(&res_permute_axes)?;
        result_size = result.size();
    }
    let res_shape = Arc::new(result.shape().inner().clone());
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
        let a_last_index = a_.ndim() - 1;
        let inner_loop_size = (if is_left {
            transposed_shape[a_.ndim() - axes.len() - 1]
        } else {
            transposed_shape[a_last_index]
        }) as usize;
        let a_size = a_.size();
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
                    res_shape.clone(),
                    result.strides().inner()
                );
                let barrier = Arc::new(Barrier::new(num_threads + 1));
                for _ in 0..num_threads {
                    let mut iterator = iterators.pop().unwrap();
                    let mut result_ptr_c = iterator.res_ptrs;
                    let mut a_data_ptr = iterator.ptrs;
                    let current_size = iterator.end - iterator.start;
                    let barrier_clone = Arc::clone(&barrier);
                    let res_shape = res_shape.clone();
                    let res_strides = result.strides().clone();
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
                        barrier_clone.wait();
                    });
                }
                barrier.wait();
            } else {
                let outer_loop_size = result_size / inner_loop_size;
                let inner_loop_size_2 = a.size() / result_size;
                if outer_loop_size == 1 {
                    let mut p = (0..a.ndim()).collect::<Vec<usize>>();
                    let front = p.remove(0);
                    p.push(front);
                    let _a = transposed_tensor.permute(&p).unwrap();
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
                    let mut slices = vec![Slice::Full; _a.ndim()];
                    let mut slices_res = vec![Slice::Full; result.ndim()];
                    let mut sliced_tensors = Vec::with_capacity(num_threads);
                    let mut sliced_res = Vec::with_capacity(num_threads);
                    let mut num_threads = 0;
                    assert_eq!(inner_loop_size, result_size);
                    for (start, end) in intervals.into_iter() {
                        if end - start == 0 {
                            continue;
                        }
                        num_threads += 1;
                        slices[(_a.ndim()) - 1] = Slice::Range((start as i64, end as i64));
                        slices_res[(result.ndim()) - 1] = Slice::Range((
                            start as i64,
                            end as i64,
                        ));
                        sliced_tensors.push(_a.slice(&slices).expect("Slice failed"));
                        sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
                    }
                    let barrier = Arc::new(Barrier::new(num_threads + 1));
                    for (inp, res) in sliced_tensors.into_iter().zip(sliced_res.into_iter()) {
                        let barrier_clone = barrier.clone();
                        pool.execute(move || {
                            res.iter_mut()
                                .zip(inp.iter())
                                .for_each(|(x, y)| {
                                    *x = op(*x, y);
                                });
                            if let Some(op3) = op3 {
                                res.iter_mut().for_each(|x| {
                                    *x = op3(*x);
                                });
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
                        res_shape.clone(),
                        result.strides().inner()
                    );
                    let barrier = Arc::new(Barrier::new(num_threads + 1));
                    for _ in (0..num_threads).rev() {
                        let mut iterator = iterators.pop().unwrap();
                        let result_ptr_c = iterator.res_ptrs;
                        let a_data_ptr = iterator.ptrs;
                        let current_size = iterator.end - iterator.start;
                        let barrier_clone = Arc::clone(&barrier);
                        let res_last_strides = *result.strides().inner().last().unwrap();
                        let res_strides = result.strides().clone();
                        let res_shape = res_shape.clone();
                        pool.execute(move || {
                            let shape_len = iterator.shape.len() as i64;
                            let mut result_ptr_c = result_ptr_c;
                            let mut a_data_ptr = a_data_ptr;
                            for _i in 0..current_size {
                                for _ in 0..inner_loop_size_2 {
                                    for i in 0..inner_loop_size as i64 {
                                        result_ptr_c[i * res_last_strides] = op(
                                            result_ptr_c[i * res_last_strides],
                                            a_data_ptr[i]
                                        );
                                    }
                                    for j in (shape_len..iterator.a_shape.len() as i64).rev() {
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
                                        result_ptr_c[i * res_last_strides] = op3(
                                            result_ptr_c[i * res_last_strides]
                                        );
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

                            barrier_clone.wait();
                        });
                    }
                    barrier.wait();
                }
            }
        });
    }
    let result = result.permute_inv(res_permute_axes)?;
    if let Some(new_shape) = new_shape {
        let result = result.reshape(new_shape)?;
        Ok(result)
    } else {
        Ok(result)
    }
}
