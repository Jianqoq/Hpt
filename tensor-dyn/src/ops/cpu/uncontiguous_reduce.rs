use std::sync::Arc;
use std::sync::Barrier;
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
use crate::ops::cpu::reduce::ReductionPreprocessor;
use crate::slice::SliceOps;
use crate::tensor_base::_Tensor;
use crate::THREAD_POOL;
use tensor_traits::TensorInfo;
use tensor_traits::ShapeManipulate;

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
        T: CommonBounds + IntoScalar<O> + tensor_types::convertion::Convertor,
        O: CommonBounds,
        F: Fn(O, T) -> O + Sync + Send + 'static + Copy,
        F2: Fn(O, O) -> O + Sync + Send + 'static + Copy,
        F3: Fn(O) -> O + Sync + Send + 'static + Copy,
        F4: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec +
            'static +
            Copy +
            std::marker::Send,
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
    transposed_axis[a.ndim() - axes.len()..].sort_by(|a, b| {
        a_.strides()[*b].cmp(&a_.strides()[*a])
    });
    transposed_axis[..a.ndim() - axes.len()].sort_by(|a, b| {
        a_.strides()[*b].cmp(&a_.strides()[*a])
    });
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
        result = _Tensor::<O, Cpu>::full(init_val, res_shape.clone())?;
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
                        res_shape.clone()
                    );
                    let barrier = Arc::new(Barrier::new(num_threads + 1));
                    for _ in (0..num_threads).rev() {
                        let mut iterator = iterators.pop().unwrap();
                        let result_ptr_c = iterator.res_ptrs;
                        let a_data_ptr = iterator.ptrs;
                        let current_size = iterator.end - iterator.start;
                        let barrier_clone = Arc::clone(&barrier);
                        pool.execute(move || {
                            let shape_len = iterator.shape.len() as i64;
                            assert_eq!(a_last_stride, 1);
                            let mut result_ptr_c = result_ptr_c;
                            let mut a_data_ptr = a_data_ptr;
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
                                    for j in (shape_len..(iterator.a_shape.len() as i64) -
                                        2).rev() {
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
    if let Some(new_shape) = new_shape {
        let result = result.reshape(new_shape)?;
        Ok(result)
    } else {
        Ok(result)
    }
}
