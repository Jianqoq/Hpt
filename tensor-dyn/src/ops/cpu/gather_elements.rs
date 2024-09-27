use std::sync::{ Arc, Barrier };

use tensor_common::shape_utils::mt_intervals;
use tensor_traits::{ CommonBounds, TensorCreator, TensorInfo };

use crate::{ backend::Cpu, tensor_base::_Tensor, THREAD_POOL };

impl<T> _Tensor<T, Cpu> where T: CommonBounds {
    /// Gathers elements from the tensor along a specified axis using given indices.
    ///
    /// This method extracts elements from the tensor along the specified `axis` based on the
    /// indices provided in the `indices` tensor. Each element in the `indices` tensor specifies
    /// which element to select along the given axis in the input tensor. The output tensor will
    /// have the same shape as the `indices` tensor.
    ///
    /// # Arguments
    ///
    /// * `indices` - A tensor of indices, specifying the positions of the elements to gather
    ///   from the input tensor along the specified axis. The size of the `indices` tensor
    ///   determines the shape of the output tensor.
    /// * `axis` - The axis along which to gather the elements. This value must be a valid axis
    ///   for the input tensor.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the gathered elements.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn gather_elements(&self, indices: &_Tensor<i64, Cpu>, axis: i64) -> anyhow::Result<Self> {
        let axis = (if axis < 0 { (self.ndim() as i64) + axis } else { axis }) as usize;
        let ret = _Tensor::<T, Cpu>::empty(indices.shape())?;
        let inner_loop_size = indices.shape()[indices.ndim() - 1] as usize;
        let outer_loop_size = indices.size() / inner_loop_size;

        THREAD_POOL.with_borrow_mut(|pool| {
            let num_threads = if outer_loop_size < pool.max_count() {
                outer_loop_size
            } else {
                pool.max_count()
            };
            let intervals = mt_intervals(outer_loop_size, num_threads);
            let mut res_ptrs = Vec::with_capacity(num_threads);
            let mut idx_ptrs = Vec::with_capacity(num_threads);
            let mut prgs = Vec::with_capacity(num_threads);
            for (start, _) in intervals.iter() {
                let mut res_ptr = ret.ptr().clone();
                let mut idx_ptr = indices.ptr().clone();
                let mut res_prg = vec![0; ret.ndim()];
                let mut amount = *start * (*ret.shape().last().unwrap() as usize);

                let mut index = 0;
                let mut idx_index = 0;
                for j in (0..ret.shape().len()).rev() {
                    res_prg[j] = (amount as i64) % ret.shape()[j];
                    amount /= ret.shape()[j] as usize;
                    index += res_prg[j] * ret.strides()[j];
                    idx_index += res_prg[j] * indices.strides()[j];
                }
                res_ptr.offset(index);
                res_ptrs.push(res_ptr);
                prgs.push(res_prg);
                idx_ptr.offset(idx_index);
                idx_ptrs.push(idx_ptr);
            }
            let idx_last_stride = indices.strides()[indices.ndim() - 1];
            let ndim = self.ndim() as i64;

            let barrier = Arc::new(Barrier::new(num_threads + 1));
            for ((((start, end), mut res_ptr), mut idx_ptr), mut prg) in intervals
                .into_iter()
                .zip(res_ptrs.into_iter())
                .zip(idx_ptrs.into_iter())
                .zip(prgs.into_iter()) {
                let shape = ret.shape().clone();
                let ret_strides = ret.strides().clone();
                let indice_strides = indices.strides().clone();
                let inp_ptr = self.ptr().clone();
                let inp_strides = self.strides().clone();
                let inp_idx_stride = self.strides()[axis];
                let inp_last_stride = self.strides()[(ndim as usize) - 1];
                let barrier_clone = barrier.clone();
                pool.execute(move || {
                    if axis == (ndim as usize) - 1 {
                        let index_cal = |prg: &[i64]| {
                            let mut acc = 0;
                            for (i, &x) in prg
                                .iter()
                                .enumerate()
                                .take((ndim as usize) - 1) {
                                acc += x * inp_strides[i];
                            }
                            acc
                        };
                        for _ in start..end {
                            for i in 0..inner_loop_size as i64 {
                                let idx = idx_ptr[i * idx_last_stride];
                                let inp_index = index_cal(&prg);
                                res_ptr[i] = inp_ptr[inp_index + idx * inp_idx_stride];
                            }
                            for j in (0..ndim - 1).rev() {
                                let j = j as usize;
                                if prg[j] < shape[j] - 1 {
                                    prg[j] += 1;
                                    res_ptr.offset(ret_strides[j]);
                                    idx_ptr.offset(indice_strides[j]);
                                    break;
                                } else {
                                    prg[j] = 0;
                                    res_ptr.offset(-ret_strides[j] * (shape[j] - 1));
                                    idx_ptr.offset(-indice_strides[j] * (shape[j] - 1));
                                }
                            }
                        }
                    } else {
                        let index_cal = |prg: &mut [i64]| {
                            let tmp = prg[axis];
                            let mut acc = 0;
                            for (i, &x) in prg
                                .iter()
                                .enumerate()
                                .take((ndim as usize) - 1) {
                                if i == axis {
                                    continue;
                                }
                                acc += x * inp_strides[i];
                            }
                            prg[axis] = tmp;
                            acc
                        };
                        let mut offset = index_cal(&mut prg);
                        for _ in start..end {
                            for i in 0..inner_loop_size as i64 {
                                let idx = idx_ptr[i * idx_last_stride];
                                res_ptr[i] = inp_ptr[
                                    offset + idx * inp_idx_stride + i * inp_last_stride
                                ];
                            }
                            for j in (0..ndim - 1).rev() {
                                let j = j as usize;
                                if prg[j] < shape[j] - 1 {
                                    prg[j] += 1;
                                    res_ptr.offset(ret_strides[j]);
                                    idx_ptr.offset(indice_strides[j]);
                                    if j != axis {
                                        offset += inp_strides[j];
                                    }
                                    break;
                                } else {
                                    prg[j] = 0;
                                    res_ptr.offset(-ret_strides[j] * (shape[j] - 1));
                                    idx_ptr.offset(-indice_strides[j] * (shape[j] - 1));
                                    if j != axis {
                                        offset -= inp_strides[j] * (shape[j] - 1);
                                    }
                                }
                            }
                        }
                    }
                    barrier_clone.wait();
                });
            }
            barrier.wait();
        });
        Ok(ret)
    }
}
