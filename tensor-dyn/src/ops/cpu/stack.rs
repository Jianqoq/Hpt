use std::sync::{ Arc, Barrier };

use tensor_common::{ shape_utils::mt_intervals, slice::Slice };
use tensor_traits::{
    shape_manipulate::ShapeManipulate,
    tensor::{ CommonBounds, TensorCreator, TensorInfo },
};

use crate::{ slice::SliceOps, tensor_base::_Tensor, THREAD_POOL };

pub(crate) fn stack<T>(
    tensors: Vec<&_Tensor<T>>,
    axis: usize,
    keepdims: bool
) -> anyhow::Result<_Tensor<T>>
    where T: CommonBounds
{
    let length = tensors.len();
    let mut all_same_shape = true;
    for i in tensors.iter() {
        tensors[0]
            .shape()
            .iter()
            .enumerate()
            .try_for_each(|(idx, x)| {
                if
                    idx != axis &&
                    i.shape().len() == tensors[0].shape().len() &&
                    *x != i.shape()[idx]
                {
                    return Err(
                        anyhow::Error::msg("Shapes except the axis to stack must be the same")
                    );
                } else if i.shape().len() != tensors[0].shape().len() {
                    return Err(
                        anyhow::Error::msg("Shape length mismatch when trying to stack tensors")
                    );
                } else if idx == axis && *x != i.shape()[idx] {
                    all_same_shape = false;
                }
                Ok(())
            })?;
    }
    let mut new_shape: Vec<i64> = vec![0;
    tensors[0].ndim()as usize];
    tensors.iter().for_each(|x| {
        new_shape[axis] += x.shape()[axis];
    });
    tensors[0]
        .shape()
        .iter()
        .enumerate()
        .for_each(|(i, x)| {
            if i != axis {
                new_shape[i] = *x;
            }
        });
    let new_tensor = _Tensor::<T>::empty(&new_shape)?;
    let mut begin = 0;
    let mut res_slices = Vec::with_capacity(length);
    for i in tensors.iter() {
        let mut selections = vec![Slice::Full; new_shape.len()];
        selections[axis] = Slice::Range((begin, begin + i.shape()[axis]));
        begin += i.shape()[axis];
        let res_tensor = new_tensor.slice(&selections)?;
        res_slices.push(res_tensor);
    }
    let tensors = tensors
        .iter()
        .map(|x| (*x).clone())
        .collect::<Vec<_Tensor<T>>>();
    THREAD_POOL.with_borrow_mut(|pool| {
        let num_threads: usize;
        if length < pool.max_count() {
            num_threads = length;
        } else {
            num_threads = pool.max_count();
        }
        let barrier = Arc::new(Barrier::new(num_threads + 1));
        let intervals: Vec<(usize, usize)> = mt_intervals(length, num_threads);
        for i in 0..num_threads {
            let barrier_clone = Arc::clone(&barrier);
            let (start, end) = intervals[i];
            let res_tensors = res_slices[start..end].to_vec();
            let inputs = tensors[start..end].to_vec();
            pool.execute(move || {
                for (res, input) in res_tensors.into_iter().zip(inputs.into_iter()) {
                    let mut res_ptr = res.ptr();
                    let mut a_data = input.ptr();
                    let a_last_stride = *input.strides().last().unwrap();
                    let inner_loop_size = *input.shape().last().unwrap();
                    let outer_loop_size = input.size() / (inner_loop_size as usize);
                    let mut prg = vec![0; input.ndim() as usize];
                    for _ in 0..outer_loop_size {
                        for i in 0..inner_loop_size {
                            let a_val = a_data[i * a_last_stride];
                            res_ptr.modify(i, a_val);
                        }
                        for j in (0..(input.ndim() as i64) - 1).rev() {
                            let j = j as usize;
                            if prg[j] < input.shape()[j] - 1 {
                                prg[j] += 1;
                                a_data.offset(input.strides()[j]);
                                res_ptr.offset(res.strides()[j]);
                                break;
                            } else {
                                prg[j] = 0;
                                a_data.offset(-(input.shape()[j] - 1) * input.strides()[j]);
                                res_ptr.offset(-(res.shape()[j] - 1) * res.strides()[j]);
                            }
                        }
                    }
                }
                barrier_clone.wait();
            });
        }
        barrier.wait();
    });
    if keepdims {
        let mut res_shape = Vec::with_capacity(new_shape.len() + 1);
        for (idx, i) in new_shape.iter().enumerate() {
            if idx == axis {
                res_shape.push(length as i64);
                res_shape.push(*i / (length as i64));
            } else {
                res_shape.push(*i);
            }
        }
        new_tensor.reshape(res_shape)
    } else {
        Ok(new_tensor)
    }
}
