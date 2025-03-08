use std::sync::Arc;

use crate::backend::Cpu;
use crate::backends::cpu::utils::binary::binary_normal::binary_fn_with_out_simd;
use crate::tensor_base::_Tensor;
use crate::THREAD_POOL;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape_utils::mt_intervals;
use hpt_common::Pointer;
use hpt_iterator::iterator_traits::ParStridedIteratorZip;
use hpt_iterator::TensorIterator;
use hpt_traits::ops::advance::{AdvancedOps, HardMax, TensorWhere};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::reduce::NormalReduce;
use hpt_traits::ops::shape_manipulate::ShapeManipulate;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::into_vec::IntoVec;
use hpt_types::type_promote::{Cmp, NormalOut, SimdCmp};

impl<T: CommonBounds + PartialOrd, const DEVICE: usize, Al> AdvancedOps
    for _Tensor<T, Cpu, DEVICE, Al>
where
    T: NormalOut<bool, Output = T> + Cast<i64>,
    f64: Cast<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;
    type Output = _Tensor<T, Cpu, DEVICE, Al>;

    type IndexOutput = _Tensor<i64, Cpu, DEVICE, Al>;

    fn pad(&self, pads: &[(i64, i64)], val: Self::Meta) -> Result<Self::Output, TensorError> {
        let res_shape = self
            .shape()
            .iter()
            .zip(pads.iter())
            .map(|(x, (a, b))| x + a + b)
            .collect::<Vec<_>>();

        let res = _Tensor::<T, Cpu, DEVICE, Al>::full(val, &res_shape)?;

        let outer_loop = self.shape()[..self.ndim() - 1].iter().product::<i64>();
        let inner_loop = self.shape()[self.ndim() - 1];
        let mut pads = pads.to_vec();
        if pads.len() < self.ndim() {
            pads.resize(self.ndim(), (0, 0));
        }
        let pads = Arc::new(pads);
        THREAD_POOL.with_borrow_mut(|pool| {
            let num_threads = if outer_loop < (pool.max_count() as i64) {
                outer_loop as usize
            } else {
                pool.max_count()
            };
            let intervals = mt_intervals(outer_loop as usize, num_threads);
            let mut ptrs = vec![];
            let mut prgs = vec![];
            for (start, _) in intervals.iter() {
                let mut ptr = self.ptr();
                let mut inp_index = 0;
                let mut inp_prg = vec![0; self.ndim()];
                let mut inp_amount = start * (self.shape()[self.ndim() - 1] as usize);
                for j in (0..self.ndim()).rev() {
                    inp_prg[j] = (inp_amount as i64) % self.shape()[j];
                    inp_amount /= self.shape()[j] as usize;
                    inp_index += inp_prg[j] * self.strides()[j];
                }
                ptr.offset(inp_index);
                ptrs.push(ptr);
                prgs.push(inp_prg);
            }
            for ((mut ptr, mut inp_prg), (start, end)) in ptrs
                .into_iter()
                .zip(prgs.into_iter())
                .zip(intervals.into_iter())
            {
                let ndim = self.ndim();
                let ts = self.strides().clone();
                let tsp = self.shape().clone();
                let rs = res.strides().clone();
                let mut res_ptr = res.ptr();
                let inp_last_stride = self.strides()[ndim - 1];
                let pads = pads.clone();
                pool.execute(move || {
                    assert_eq!(tsp.len(), ndim);
                    assert_eq!(ts.len(), ndim);
                    assert_eq!(rs.len(), ndim);
                    for (dim_idx, (&prg_idx, &stride)) in inp_prg.iter().zip(rs.iter()).enumerate()
                    {
                        let padded_idx = prg_idx + pads[dim_idx].0;
                        let offset = padded_idx * stride;
                        res_ptr.offset(offset);
                    }
                    for _ in start..end {
                        for i in 0..inner_loop {
                            res_ptr[i] = ptr[i * inp_last_stride];
                        }
                        for j in (0..ndim - 1).rev() {
                            let j = j;
                            if inp_prg[j] < tsp[j] - 1 {
                                inp_prg[j] += 1;
                                ptr.offset(ts[j]);
                                res_ptr.offset(rs[j]);
                                break;
                            } else {
                                inp_prg[j] = 0;
                                ptr.offset(-ts[j] * (tsp[j] - 1));
                                res_ptr.offset(-rs[j] * (tsp[j] - 1));
                            }
                        }
                    }
                });
            }
            pool.join();
        });

        Ok(res)
    }

    fn topk(
        &self,
        k: i64,
        dim: i64,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self::IndexOutput, Self::Output), TensorError> {
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
        let res = _Tensor::<T, Cpu, DEVICE, Al>::empty(&res_shape)?;
        let res_indices = _Tensor::<i64, Cpu, DEVICE, Al>::empty(&res_shape)?;
        let transposed_res = res.permute(&axes)?;
        let transposed_res_indices = res_indices.permute(&axes)?;
        let inner_loop = *transposed.shape().last().unwrap() as isize;
        let outer_loop_size = transposed.size() / inner_loop as usize;

        let res_ptr = transposed_res.ptr();
        let res_indices_ptr = transposed_res_indices.ptr();
        let transposed_res_shape = transposed_res.shape();
        let transposed_res_strides = transposed_res.strides();
        THREAD_POOL.with_borrow_mut(move |x| {
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
            if k > inner_loop as i64 {
                return Err::<(), TensorError>(
                    ShapeError::TopkError {
                        message: format!(
                            "k ({}) cannot be larger than inner_loop size ({})",
                            k, inner_loop
                        ),
                        location: caller,
                    }
                    .into(),
                );
            }

            if k <= 0 {
                return Err::<(), TensorError>(
                    ShapeError::TopkError {
                        message: format!("k ({}) must be positive", k),
                        location: caller,
                    }
                    .into(),
                );
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
                    let data = &mut vec![(T::ZERO, 0); inner_loop as usize];
                    for _ in start..end {
                        let data = fill_buffer(data, ptr.clone(), inner_loop as i64, tls as i64);
                        let before = if largest {
                            if k as usize == data.len() {
                                data.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
                                data
                            } else {
                                data.select_nth_unstable_by(k as usize, |x, y| {
                                    y.0.partial_cmp(&x.0).unwrap()
                                })
                                .0
                            }
                        } else {
                            if k as usize == data.len() {
                                data.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
                                data
                            } else {
                                data.select_nth_unstable_by(k as usize, |x, y| {
                                    x.0.partial_cmp(&y.0).unwrap()
                                })
                                .0
                            }
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

    fn onehot(
        &self,
        depth: usize,
        mut axis: i64,
        true_val: Self::Meta,
        false_val: Self::Meta,
    ) -> Result<Self::Output, TensorError> {
        let mut new_shape = self.shape().inner().clone();
        if axis < 0 {
            axis += self.ndim() as i64;
        }
        ShapeError::check_index_out_of_range(axis, self.ndim() as i64)?;
        axis += 1;
        new_shape.insert(axis as usize, depth as i64);
        let res = _Tensor::<T, Cpu, DEVICE, Al>::full(false_val, new_shape)?;
        let mut permute_axes = (0..res.ndim()).collect::<Vec<usize>>();
        permute_axes.retain(|x| *x != (axis as usize));
        permute_axes.push(axis as usize);
        let permuted_res = res.permute(permute_axes)?;
        assert_eq!(
            &permuted_res.shape()[..res.ndim() - 1],
            self.shape().inner().as_slice()
        );
        let inner_loop_size = *permuted_res.shape().last().unwrap();
        let outer_loop_size = permuted_res.size() as i64 / inner_loop_size;

        let last_strides = *permuted_res.strides().last().unwrap();

        THREAD_POOL.with_borrow_mut(|pool| {
            let num_threads = if outer_loop_size < (pool.max_count() as i64) {
                outer_loop_size as usize
            } else {
                pool.max_count()
            };
            let intervals = mt_intervals(outer_loop_size as usize, num_threads);
            for (start, end) in intervals.into_iter() {
                let mut res_ptr = permuted_res.ptr();
                let inp_shape = self.shape().clone();
                let inp_strides = self.strides().clone();
                let permuted_res_shape = permuted_res.shape().clone();
                let permuted_res_strides = permuted_res.strides().clone();
                let inp_ptr = self.ptr();
                pool.execute(move || {
                    for i in start..end {
                        let mut amount = i;
                        let mut res_amount = i as i64 * inner_loop_size;
                        let mut index = 0i64;
                        let mut res_index = 0i64;
                        for j in (0..inp_shape.len()).rev() {
                            index += ((amount as i64) % inp_shape[j]) * inp_strides[j];
                            amount /= inp_shape[j] as usize;
                        }
                        for j in (0..permuted_res_shape.len()).rev() {
                            res_index += ((res_amount as i64) % permuted_res_shape[j])
                                * permuted_res_strides[j];
                            res_amount /= permuted_res_shape[j];
                        }
                        let dp: i64 = inp_ptr[index].cast();
                        res_ptr[res_index + dp * last_strides] = true_val;
                    }
                });
            }
            pool.join();
        });

        Ok(res)
    }

    // fn gather(&self, indices: &Self::IndexOutput, axis: i64) -> Result<Self::Output, TensorError> {
    //     if self.ndim() != indices.ndim() {
    //         ShapeError::check_ndim_enough(
    //             "gather requires input and indices have the same number of dimensions.".to_string(),
    //             self.ndim(),
    //             indices.ndim(),
    //         )?;
    //     }

    //     let axis = if axis < 0 {
    //         self.ndim() as i64 + axis
    //     } else {
    //         axis
    //     } as usize;

    //     let mut permute_axes = (0..indices.ndim()).collect::<Vec<usize>>();
    //     permute_axes.retain(|x| *x != (axis as usize));
    //     permute_axes.push(axis as usize);
    //     let permuted_indices = indices.permute(&permute_axes)?;
    //     let inner_size = *permuted_indices.shape().last().unwrap();
    //     let outer_size = permuted_indices.size() as i64 / inner_size;

    //     let mut res_shape = self.shape().to_vec();
    //     res_shape[axis] = *permuted_indices.shape().last().unwrap();
    //     let res = Self::empty(res_shape)?;
    //     let permuted_self = self.permute(&permute_axes)?;

    //     if self.ndim() == 1 {
    //         for i in 0..indices.size() {
    //             let idx = indices.ptr()[i];
    //             if idx < 0 || idx >= self.shape()[0] as i64 {
    //                 return Err(ShapeError::DimOutOfRange {
    //                     expected: 0..self.shape()[0] as i64,
    //                     actual: idx,
    //                     location: core::panic::Location::caller(),
    //                 }
    //                 .into());
    //             }
    //             res.ptr()[i] = self.ptr()[idx as usize];
    //         }
    //         return Ok(res);
    //     }

    //     THREAD_POOL.with_borrow_mut(|pool| {
    //         let num_threads = (outer_size as usize).min(pool.max_count());
    //         let intervals = mt_intervals(outer_size as usize, num_threads);

    //         for (start, end) in intervals {
    //             let self_ptr = permuted_self.ptr();
    //             let indices_ptr = permuted_indices.ptr();
    //             let mut res_ptr = res.ptr();

    //             let gather_size = indices.shape()[axis];
    //             let self_last_strides = *self.strides().last().unwrap();
    //             let axis_self_last_stride = self.strides()[axis];
    //             let self_inner_size = *permuted_self.shape().last().unwrap();

    //             let res_shape = res.shape().clone();
    //             let res_strides = res.strides().clone();

    //             let permuted_self_shape = permuted_self.shape().clone();
    //             let permuted_self_strides = permuted_self.strides().clone();

    //             let indices_shape = indices.shape().clone();
    //             let indices_strides = indices.strides().clone();

    //             pool.execute(move || {
    //                 for outer_idx in start as i64..end as i64 {
    //                     let mut indices_amount = outer_idx * *indices_shape.last().unwrap();
    //                     let mut indices_offset = 0;
    //                     for j in (0..indices_shape.len()).rev() {
    //                         indices_offset +=
    //                             (indices_amount % indices_shape[j]) * indices_strides[j];
    //                         indices_amount /= indices_shape[j];
    //                     }

    //                     let mut res_indices_amount = outer_idx * gather_size;
    //                     let mut res_indices_offset = 0;
    //                     for j in (0..res_shape.len()).rev() {
    //                         res_indices_offset +=
    //                             (res_indices_amount % res_shape[j]) * res_strides[j];
    //                         res_indices_amount /= res_shape[j];
    //                     }

    //                     let mut inp_indices_amount = outer_idx * self_inner_size;
    //                     let mut inp_indices_offset = 0;
    //                     for j in (0..permuted_self_shape.len() - 1).rev() {
    //                         inp_indices_offset += (inp_indices_amount % permuted_self_shape[j])
    //                             * permuted_self_strides[j];
    //                         inp_indices_amount /= permuted_self_shape[j];
    //                     }
    //                     for gather_idx in 0..gather_size {
    //                         let index = indices_ptr[indices_offset + gather_idx];
    //                         let val = self_ptr[(inp_indices_offset
    //                             + gather_idx * self_last_strides)
    //                             + index * axis_self_last_stride];
    //                         res_ptr[res_indices_offset + gather_idx] = val;
    //                     }
    //                 }
    //             });
    //         }
    //         pool.join();
    //     });

    //     Ok(res)
    // }

    // fn gather_elements(
    //     &self,
    //     indices: &Self::IndexOutput,
    //     axis: i64,
    // ) -> Result<Self::Output, TensorError> {
    //     let axis = (if axis < 0 {
    //         (self.ndim() as i64) + axis
    //     } else {
    //         axis
    //     }) as usize;
    //     let ret = _Tensor::<T, Cpu, DEVICE>::empty(indices.shape())?;
    //     let inner_loop_size = indices.shape()[indices.ndim() - 1] as usize;
    //     let outer_loop_size = indices.size() / inner_loop_size;

    //     THREAD_POOL.with_borrow_mut(|pool| {
    //         let num_threads = if outer_loop_size < pool.max_count() {
    //             outer_loop_size
    //         } else {
    //             pool.max_count()
    //         };
    //         let intervals = mt_intervals(outer_loop_size, num_threads);
    //         let mut res_ptrs = Vec::with_capacity(num_threads);
    //         let mut idx_ptrs = Vec::with_capacity(num_threads);
    //         let mut prgs = Vec::with_capacity(num_threads);
    //         for (start, _) in intervals.iter() {
    //             let mut res_ptr = ret.ptr().clone();
    //             let mut idx_ptr = indices.ptr().clone();
    //             let mut res_prg = vec![0; ret.ndim()];
    //             let mut amount = *start * (*ret.shape().last().unwrap() as usize);

    //             let mut index = 0;
    //             let mut idx_index = 0;
    //             for j in (0..ret.shape().len()).rev() {
    //                 res_prg[j] = (amount as i64) % ret.shape()[j];
    //                 amount /= ret.shape()[j] as usize;
    //                 index += res_prg[j] * ret.strides()[j];
    //                 idx_index += res_prg[j] * indices.strides()[j];
    //             }
    //             res_ptr.offset(index);
    //             res_ptrs.push(res_ptr);
    //             prgs.push(res_prg);
    //             idx_ptr.offset(idx_index);
    //             idx_ptrs.push(idx_ptr);
    //         }
    //         let idx_last_stride = indices.strides()[indices.ndim() - 1];
    //         let ndim = self.ndim() as i64;

    //         for ((((start, end), mut res_ptr), mut idx_ptr), mut prg) in intervals
    //             .into_iter()
    //             .zip(res_ptrs.into_iter())
    //             .zip(idx_ptrs.into_iter())
    //             .zip(prgs.into_iter())
    //         {
    //             let shape = ret.shape().clone();
    //             let ret_strides = ret.strides().clone();
    //             let indice_strides = indices.strides().clone();
    //             let inp_ptr = self.ptr().clone();
    //             let inp_strides = self.strides().clone();
    //             let inp_idx_stride = self.strides()[axis];
    //             let inp_last_stride = self.strides()[(ndim as usize) - 1];
    //             pool.execute(move || {
    //                 if axis == (ndim as usize) - 1 {
    //                     let index_cal = |prg: &[i64]| {
    //                         let mut acc = 0;
    //                         for (i, &x) in prg.iter().enumerate().take((ndim as usize) - 1) {
    //                             acc += x * inp_strides[i];
    //                         }
    //                         acc
    //                     };
    //                     for _ in start..end {
    //                         for i in 0..inner_loop_size as i64 {
    //                             let idx = idx_ptr[i * idx_last_stride];
    //                             let inp_index = index_cal(&prg);
    //                             res_ptr[i] = inp_ptr[inp_index + idx * inp_idx_stride];
    //                         }
    //                         for j in (0..ndim - 1).rev() {
    //                             let j = j as usize;
    //                             if prg[j] < shape[j] - 1 {
    //                                 prg[j] += 1;
    //                                 res_ptr.offset(ret_strides[j]);
    //                                 idx_ptr.offset(indice_strides[j]);
    //                                 break;
    //                             } else {
    //                                 prg[j] = 0;
    //                                 res_ptr.offset(-ret_strides[j] * (shape[j] - 1));
    //                                 idx_ptr.offset(-indice_strides[j] * (shape[j] - 1));
    //                             }
    //                         }
    //                     }
    //                 } else {
    //                     let index_cal = |prg: &mut [i64]| {
    //                         let tmp = prg[axis];
    //                         let mut acc = 0;
    //                         for (i, &x) in prg.iter().enumerate().take((ndim as usize) - 1) {
    //                             if i == axis {
    //                                 continue;
    //                             }
    //                             acc += x * inp_strides[i];
    //                         }
    //                         prg[axis] = tmp;
    //                         acc
    //                     };
    //                     let mut offset = index_cal(&mut prg);
    //                     for _ in start..end {
    //                         for i in 0..inner_loop_size as i64 {
    //                             let idx = idx_ptr[i * idx_last_stride];
    //                             res_ptr[i] =
    //                                 inp_ptr[offset + idx * inp_idx_stride + i * inp_last_stride];
    //                         }
    //                         for j in (0..ndim - 1).rev() {
    //                             let j = j as usize;
    //                             if prg[j] < shape[j] - 1 {
    //                                 prg[j] += 1;
    //                                 res_ptr.offset(ret_strides[j]);
    //                                 idx_ptr.offset(indice_strides[j]);
    //                                 if j != axis {
    //                                     offset += inp_strides[j];
    //                                 }
    //                                 break;
    //                             } else {
    //                                 prg[j] = 0;
    //                                 res_ptr.offset(-ret_strides[j] * (shape[j] - 1));
    //                                 idx_ptr.offset(-indice_strides[j] * (shape[j] - 1));
    //                                 if j != axis {
    //                                     offset -= inp_strides[j] * (shape[j] - 1);
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             });
    //         }
    //         pool.join();
    //     });
    //     Ok(ret)
    // }

    fn scatter(
        &self,
        indices: &Self::IndexOutput,
        axis: i64,
        src: &Self::Output,
    ) -> Result<Self::Output, TensorError> {
        let axis = if axis < 0 {
            self.ndim() as i64 + axis
        } else {
            axis
        } as usize;
        let result = self.clone();

        let ndim = self.ndim();
        let shape = self.shape();

        let mut position = vec![0; ndim];
        for i in 0..indices.size() as i64 {
            let mut tmp = i;
            for d in (0..ndim).rev() {
                position[d] = tmp % shape[d];
                tmp /= shape[d];
            }

            let target_idx = indices.ptr()[i] as i64;
            let src_val = src.ptr()[i];

            let mut target_offset = 0i64;
            for d in 0..ndim {
                if d == axis {
                    target_offset += target_idx * result.strides()[d];
                } else {
                    target_offset += position[d] * result.strides()[d];
                }
            }
            result.ptr()[target_offset] = src_val;
        }

        Ok(result)
    }
}

impl<T, const DEVICE: usize, Al> HardMax<T> for _Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Cmp<Output = bool>,
    <T as TypeCommon>::Vec: SimdCmp,
    <T::Vec as SimdCmp>::Output: IntoVec<T::Vec>,
    bool: NormalOut<T> + Cast<T>,
    Al: Allocator + 'static + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, Al>;
    fn hardmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        let axis = (if axis < 0 {
            (self.layout.ndim() as i64) + axis
        } else {
            axis
        }) as usize;
        let max = NormalReduce::<T>::max(self, axis as i64, true)?;
        let ret = {
            use hpt_types::traits::VecTrait;
            binary_fn_with_out_simd(
                self,
                &max,
                |a, max| a._eq(max).cast()._mul(T::ONE),
                |a, max| {
                    let one = T::Vec::splat(T::ONE);
                    a._eq(max).into_vec()._mul(one)
                },
                None::<Self::Output>,
            )?
        };
        Ok(ret)
    }
}

impl<T, const DEVICE: usize, Al> TensorWhere for _Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, Al>;
    type Condition = _Tensor<bool, Cpu, DEVICE, Al>;
    fn tensor_where(
        condition: &Self::Condition,
        x: &Self::Output,
        y: &Self::Output,
    ) -> Result<Self::Output, TensorError> {
        Ok(condition
            .par_iter()
            .zip(x.par_iter())
            .zip(y.par_iter())
            .strided_map(|(res, ((condition, x), y))| {
                *res = if condition { x } else { y };
            })
            .collect())
    }
}
