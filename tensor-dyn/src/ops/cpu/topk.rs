#![allow(unused)]

use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::Barrier;

use crate::tensor::Tensor;
use crate::tensor_base::_Tensor;
use crate::THREAD_POOL;
use rand::Rng;
use tensor_common::error::base::TensorError;
use tensor_common::error::shape::ShapeError;
use tensor_common::utils::pointer::Pointer;
use tensor_common::shape::shape_utils::mt_intervals;
use tensor_traits::CommonBounds;
use tensor_traits::ShapeManipulate;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;

fn fill_buffer<T: Copy>(
    data: &mut Vec<(T, usize)>,
    ptr: Pointer<T>,
    inner_loop: i64,
    tls: i64,
) -> &mut Vec<(T, usize)> {
    let mut i = 0;
    if tls != 1 {
        assert_eq!(data.len(), inner_loop as usize);
        while i < inner_loop {
            data[i as usize] = (ptr[i * tls], i as usize);
            i += 1;
        }
    } else {
        assert_eq!(data.len(), inner_loop as usize);
        while i < inner_loop {
            data[i as usize] = (ptr[i], i as usize);
            i += 1;
        }
    }
    data
}

impl<T> _Tensor<T>
where
    T: CommonBounds + PartialOrd,
{
    /// Returns the top `k` values and their indices along the specified dimension.
    ///
    /// This method retrieves the top `k` largest or smallest values from the tensor along the given `dim` axis,
    /// and returns two tensors: one containing the values and the other containing the corresponding indices.
    /// You can specify whether to retrieve the largest or smallest values and whether the result should be sorted.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of top elements to retrieve.
    /// * `dim` - The dimension along which to find the top `k` elements.
    /// * `largest` - A boolean flag indicating whether to retrieve the largest (`true`) or smallest (`false`) values.
    /// * `sorted` - A boolean flag indicating whether the result should be sorted in descending order (if `largest` is `true`) or ascending order (if `largest` is `false`).
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tuple of two tensors:
    /// * The first tensor contains the indices of the top `k` values.
    /// * The second tensor contains the corresponding top `k` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn topk(
        &self,
        k: i64,
        dim: i64,
        largest: bool,
        sorted: bool,
    ) -> Result<(_Tensor<i64>, _Tensor<T>), TensorError> {
        let caller = core::panic::Location::caller();
        let mut axes = (0..self.ndim() as i64).collect::<Vec<i64>>();
        axes.swap(dim as usize, self.ndim() - 1);
        let transposed = self.permute(&axes)?;
        let res_shape = self
            .shape()
            .iter()
            .enumerate()
            .map(|(idx, x)| if idx == (dim as usize) { k } else { *x })
            .collect::<Vec<i64>>();
        let res = _Tensor::<T>::empty(&res_shape)?;
        let res_indices = _Tensor::<i64>::empty(&res_shape)?;
        let transposed_res = res.permute(&axes)?;
        let transposed_res_indices = res_indices.permute(&axes)?;

        let outer_loop_size = self.size() / (*self.shape().last().unwrap() as usize);

        let res_ptr = transposed_res.ptr();
        let res_indices_ptr = transposed_res_indices.ptr();
        let transposed_res_shape = transposed_res.shape();
        let transposed_res_strides = transposed_res.strides();
        let status = THREAD_POOL.with_borrow_mut(move |x| {
            let num_threads = if outer_loop_size < x.max_count() {
                outer_loop_size
            } else {
                x.max_count()
            };
            let intervals = mt_intervals(outer_loop_size, num_threads);

            let mut prgs = vec![];
            let mut ptrs = vec![];
            let mut res_ptrs = vec![];
            let mut res_indices_ptrs = vec![];
            let mut res_prgs = vec![];

            for (start, _) in intervals.iter() {
                let mut ptr = transposed.ptr();
                let mut res_ptr_cpy = res_ptr.clone();
                let mut res_indices_ptr_cpy = res_indices_ptr.clone();
                let mut current_prg = vec![0; transposed.ndim()];
                let mut res_shape_prg = vec![0; transposed_res_shape.len()];
                let mut amount = start * (*transposed.shape().last().unwrap() as usize);
                let mut res_amount = start * (*transposed_res_shape.last().unwrap() as usize);
                let mut index = 0;
                let mut res_index = 0;
                for j in (0..transposed.ndim()).rev() {
                    current_prg[j] = (amount as i64) % transposed.shape()[j];
                    res_shape_prg[j] = (res_amount as i64) % transposed_res_shape[j];
                    amount /= transposed.shape()[j] as usize;
                    res_amount /= transposed_res_shape[j] as usize;
                    index += current_prg[j] * transposed.strides()[j];
                    res_index += res_shape_prg[j] * transposed_res_strides[j];
                }
                ptr.offset(index);
                prgs.push(current_prg);
                ptrs.push(ptr);
                res_ptr_cpy.offset(res_index);
                res_indices_ptr_cpy.offset(res_index);
                res_ptrs.push(res_ptr_cpy);
                res_indices_ptrs.push(res_indices_ptr_cpy);
                res_prgs.push(res_shape_prg);
            }
            let inner_loop = *transposed.shape().last().unwrap() as isize;
            if inner_loop < (k as isize) {
                return Err::<(), TensorError>(ShapeError::TopkError {
                    message: format!("k {} larger than inner loop size {}", k, inner_loop),
                    location: caller,
                }
                .into());
            }
            for (
                (((((start, end), mut prg), mut res_prg), mut ptr), mut res_ptr),
                mut res_indices_ptr,
            ) in intervals
                .into_iter()
                .rev()
                .zip(prgs.into_iter().rev())
                .zip(res_prgs.into_iter().rev())
                .zip(ptrs.into_iter().rev())
                .zip(res_ptrs.into_iter().rev())
                .zip(res_indices_ptrs.into_iter().rev())
            {
                let res_inner_loop = *transposed_res_shape.last().unwrap() as isize;
                let tls = *transposed.strides().last().unwrap() as isize;
                let rls = *transposed_res_strides.last().unwrap() as isize;
                let ndim = transposed.ndim() as i64;
                let ts = transposed.strides().clone();
                let rts = transposed_res_strides.clone();
                let tsp = transposed.shape().clone();
                let rtsp = transposed_res_shape.clone();
                x.execute(move || {
                    let mut data = &mut vec![(T::ZERO, 0); inner_loop as usize];
                    for _ in start..end {
                        let data = fill_buffer(data, ptr.clone(), inner_loop as i64, tls as i64);
                        let (before, _, _) = if largest {
                            data.select_nth_unstable_by(k as usize, |x, y| {
                                y.0.partial_cmp(&x.0).unwrap()
                            })
                        } else {
                            data.select_nth_unstable_by(k as usize, |x, y| {
                                x.0.partial_cmp(&y.0).unwrap()
                            })
                        };
                        if sorted {
                            if largest {
                                before.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
                            } else {
                                before.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
                            }
                        }
                        for i in 0..res_inner_loop {
                            let (val, idx) = before[i as usize];
                            res_ptr[i * rls] = val;
                            res_indices_ptr[i * rls] = idx as i64;
                        }
                        for j in (0..ndim - 1).rev() {
                            let j = j as usize;
                            if prg[j] < tsp[j] - 1 {
                                prg[j] += 1;
                                ptr += ts[j];
                                break;
                            } else {
                                prg[j] = 0;
                                ptr -= ts[j] * (tsp[j] - 1);
                            }
                        }
                        for j in (0..ndim - 1).rev() {
                            let j = j as usize;
                            if res_prg[j] < rtsp[j] - 1 {
                                res_prg[j] += 1;
                                res_ptr += rts[j];
                                res_indices_ptr += rts[j];
                                break;
                            } else {
                                res_prg[j] = 0;
                                res_ptr -= rts[j] * (rtsp[j] - 1);
                                res_indices_ptr -= rts[j] * (rtsp[j] - 1);
                            }
                        }
                    }
                });
            }
            x.join();
            Ok(())
        })?;
        Ok((
            transposed_res_indices.permute(&axes)?,
            transposed_res.permute(&axes)?,
        ))
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds + PartialOrd,
{
    /// Returns the top `k` largest or smallest elements along a specified dimension of the tensor.
    ///
    /// The `topk` function returns the top `k` elements along the specified dimension of the tensor, where `largest` determines
    /// whether to return the largest or smallest elements. The result can also be sorted in descending order if `sorted` is `true`.
    ///
    /// # Parameters
    ///
    /// - `k`: The number of top elements to return.
    /// - `dim`: The dimension along which to select the top `k` elements.
    /// - `largest`: If `true`, returns the largest elements. If `false`, returns the smallest elements.
    /// - `sorted`: If `true`, sorts the returned elements in descending order.
    ///
    /// # Returns
    ///
    /// - A tuple of two tensors: one containing the top `k` elements and another containing the corresponding indices.
    pub fn topk(
        &self,
        k: i64,
        dim: i64,
        largest: bool,
        sorted: bool,
    ) -> anyhow::Result<(Tensor<i64>, Tensor<T>)> {
        let (a, b) = self.inner.topk(k, dim, largest, sorted)?;
        Ok((a.into(), b.into()))
    }
}

fn topk_with_indices<T, F>(arr: &mut [(T, usize)], k: usize, compare: F)
where
    T: PartialOrd + Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let n = arr.len();
    quickselect_with_indices(arr, 0, n - 1, n - k, compare);
}

fn quickselect_with_indices<T, F>(
    arr: &mut [(T, usize)],
    low: usize,
    high: usize,
    k: usize,
    compare: F,
) where
    T: PartialOrd + Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    if low < high {
        let pi = partition_with_indices(arr, low, high, compare);
        if pi > k {
            quickselect_with_indices(arr, low, pi - 1, k, compare);
        } else if pi < k {
            quickselect_with_indices(arr, pi + 1, high, k, compare);
        }
    }
}

fn partition_with_indices<T, F>(
    arr: &mut [(T, usize)],
    low: usize,
    high: usize,
    compare: F,
) -> usize
where
    T: PartialOrd + Copy,
    F: Fn(&T, &T) -> Ordering + Copy,
{
    let mut rng = rand::thread_rng();
    let pivot_index = rng.gen_range(low..=high);
    arr.swap(pivot_index, high);
    let pivot = arr[high].0;
    let mut i = low;

    for j in low..high {
        if compare(&arr[j].0, &pivot) != Ordering::Greater {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, high);
    i
}
