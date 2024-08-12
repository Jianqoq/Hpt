use crate::backend::Cpu;
use crate::tensor_base::_Tensor;
use crate::THREAD_POOL;
use std::{ ops::Mul, sync::{ Arc, Barrier } };
use num::traits::MulAdd;
use tensor_common::{ pointer::Pointer, shape_utils::mt_intervals };
use tensor_traits::{ CommonBounds, TensorCreator, TensorInfo };
use tensor_types::type_promote::{ FloatOut, NormalOut };
use tensor_common::shape::Shape;
use tensor_types::into_scalar::IntoScalar;

use super::unary::FloatType;

impl<T> _Tensor<T, Cpu>
    where
        T: CommonBounds +
            NormalOut<Output = T> +
            Mul<Output = T> +
            std::ops::AddAssign +
            MulAdd<Output = T> +
            FloatOut,
        i64: IntoScalar<T>,
        FloatType<T>: CommonBounds
{
    pub fn avgpool<S>(
        &self,
        kernel_shape: S,
        steps: Option<&[i64]>,
        pads: Option<&[(i64, i64)]>,
        dilation: Option<&[i64]>
    ) -> anyhow::Result<_Tensor<FloatType<T>, Cpu>>
        where S: Into<Shape>
    {
        let _kernel_shape = kernel_shape.into();
        let process_pads = |pads: Option<&[(i64, i64)]>| {
            if let Some(_pads) = pads {
                _pads
                    .iter()
                    .map(|(a, b)| (*a, *b))
                    .collect::<Vec<_>>()
            } else {
                vec![(0, 0); _kernel_shape.len()]
            }
        };
        let _pads = process_pads(pads);
        let _steps = if let Some(_steps) = steps {
            _steps.to_vec()
        } else {
            vec![1; _kernel_shape.len()]
        };
        let _dilation = if let Some(_dilation) = dilation {
            Arc::new(
                _dilation
                    .iter()
                    .map(|a| *a)
                    .collect::<Vec<_>>()
            )
        } else {
            Arc::new(vec![1; _kernel_shape.len()])
        };
        let mut loop_shape = vec![];
        loop_shape.push(self.shape()[0]);
        loop_shape.push(self.shape()[1]);
        assert_eq!(self.shape().len(), _kernel_shape.len() + 2);
        self.shape()
            .iter()
            .skip(2)
            .zip(_kernel_shape.iter())
            .enumerate()
            .for_each(|(idx, (x, y))| {
                let (p_begin, p_end) = _pads[idx];
                let o = (*x + p_begin + p_end - _dilation[idx] * (*y - 1) - 1) / _steps[idx] + 1;
                loop_shape.push(o);
            });
        let loop_shape = Arc::new(loop_shape);
        let ret = _Tensor::<FloatType<T>, Cpu>::empty(&loop_shape)?;

        let outer_loop_size = loop_shape.iter().product::<i64>() / loop_shape.last().unwrap();
        let inner_loop_size = *loop_shape.last().unwrap();
        let kernel_outer_loop_size =
            _kernel_shape.iter().product::<i64>() / _kernel_shape.last().unwrap();
        let kernel_inner_loop_size = *_kernel_shape.last().unwrap();
        let kernel_ndim = _kernel_shape.len();
        let kernal_shape = Arc::new(_kernel_shape);
        let kernel_size: T = kernal_shape.iter().product::<i64>().into_scalar();
        THREAD_POOL.with_borrow_mut(|pool| {
            let num_threads = if (outer_loop_size as usize) < pool.max_count() {
                outer_loop_size as usize
            } else {
                pool.max_count()
            };
            let intervals = mt_intervals(outer_loop_size as usize, num_threads);
            let mut prgs = vec![];
            let mut res_ptrs = vec![];
            let mut ptrs = vec![];
            let mut padding_bounds_rights = vec![];
            let mut padding_bounds_lefts = vec![];

            for (start, _) in intervals.iter() {
                let mut amount = *start * (inner_loop_size as usize);
                let mut current_prg: Vec<i64> = vec![0; loop_shape.len()];
                let mut res_offset = 0;
                let mut inp_offset = 0;
                let mut padding_bounds_right = (0..kernel_ndim)
                    .map(|x| self.shape()[2..][x] + _pads[x].0)
                    .collect::<Vec<_>>();
                let mut padding_bounds_left = (0..kernel_ndim)
                    .map(|x| _pads[x].0)
                    .collect::<Vec<_>>();
                // inp[batch * inp.strides[0] + g * in_channel * inp.strides[1] + c * inp.strides[1] + ...]
                for j in (0..loop_shape.len()).rev() {
                    current_prg[j] = (amount as i64) % loop_shape[j];
                    amount /= loop_shape[j] as usize;
                    inp_offset += current_prg[j] * self.strides()[j];
                    res_offset += current_prg[j] * ret.strides()[j];
                    if j >= 2 {
                        padding_bounds_right[j - 2] -= current_prg[j] * _steps[j - 2];
                        padding_bounds_left[j - 2] -= current_prg[j] * _steps[j - 2];
                    }
                }
                let mut inp_ptr = self.ptr();
                let mut res_ptr = ret.ptr();
                inp_ptr.offset(inp_offset);
                res_ptr.offset(res_offset);
                prgs.push(current_prg);
                ptrs.push(inp_ptr);
                res_ptrs.push(res_ptr);
                padding_bounds_rights.push(padding_bounds_right);
                padding_bounds_lefts.push(padding_bounds_left);
            }
            let barrier = Arc::new(Barrier::new(num_threads + 1));
            for (
                (((((start, end), mut prg), mut inp_ptr), mut res_ptr), mut padding_bounds_right),
                mut padding_bounds_left,
            ) in intervals
                .into_iter()
                .zip(prgs.into_iter())
                .zip(ptrs.into_iter())
                .zip(res_ptrs.into_iter())
                .zip(padding_bounds_rights.into_iter())
                .zip(padding_bounds_lefts.into_iter()) {
                let barrier_clone = barrier.clone();
                let mut reduce_prg = vec![0; kernel_ndim];
                let inp_strides = self.strides().clone();
                let _kernel_shape = kernal_shape.clone();
                let dilations = _dilation.clone();
                let ret_strides = ret.strides().clone();
                let ret_last_stride = *ret.strides().last().unwrap();
                let loop_shape = loop_shape.clone();
                let last_steps = *_steps.last().unwrap();
                let steps = _steps.clone();
                let cache = *inp_strides.last().unwrap() * *dilations.last().unwrap();
                let pads = _pads.clone();
                let inp_shape = self.shape().clone(); // [batch, in_channel, ...]
                let pad_offset = (0..kernel_ndim)
                    .map(|x| pads[x].0 * inp_strides[2..][x])
                    .sum::<i64>();
                pool.execute(move || {
                    let inp_reduce_strides = &inp_strides[2..];
                    for _ in start..end {
                        for x in 0..inner_loop_size {
                            let begin = x * last_steps;
                            let sum = _sum(
                                begin as isize,
                                kernel_outer_loop_size as isize,
                                kernel_inner_loop_size as isize,
                                &mut inp_ptr,
                                &_kernel_shape,
                                &dilations,
                                &mut reduce_prg,
                                kernel_ndim,
                                inp_reduce_strides,
                                cache as isize,
                                &padding_bounds_right,
                                &padding_bounds_left,
                                pad_offset as isize
                            );
                            res_ptr.modify(x * ret_last_stride, sum._div(kernel_size));
                            padding_bounds_right[kernel_ndim - 1] -= steps.last().unwrap();
                            padding_bounds_left[kernel_ndim - 1] -= steps.last().unwrap();
                        }
                        padding_bounds_right[kernel_ndim - 1] =
                            pads[kernel_ndim - 1].0 + inp_shape[2..][kernel_ndim - 1];
                        padding_bounds_left[kernel_ndim - 1] = pads[kernel_ndim - 1].0;
                        for k in (0..loop_shape.len() - 1).rev() {
                            if prg[k] < loop_shape[k] - 1 {
                                prg[k] += 1;
                                inp_ptr.offset(inp_strides[k]);
                                res_ptr.offset(ret_strides[k]);
                                if k >= 2 {
                                    padding_bounds_right[k - 2] -= steps[k - 2];
                                    padding_bounds_left[k - 2] -= steps[k - 2];
                                }
                                break;
                            } else {
                                prg[k] = 0;
                                inp_ptr.offset(-inp_strides[k] * (loop_shape[k] - 1));
                                res_ptr.offset(-ret_strides[k] * (loop_shape[k] - 1));
                                if k >= 2 {
                                    padding_bounds_right[k - 2] =
                                        pads[k - 2].0 + inp_shape[2..][k - 2];
                                    padding_bounds_left[k - 2] = pads[k - 2].0;
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

#[inline]
fn _sum<T>(
    begin: isize,
    outer: isize,
    inner: isize,
    inp_ptr: &mut Pointer<T>,
    _kernel_shape: &[i64],
    dilations: &[i64],
    reduce_prg: &mut Vec<i64>,
    kernel_ndim: usize,
    inp_reduce_strides: &[i64],
    cache: isize,
    padding_bounds_right: &[i64],
    padding_bounds_left: &[i64],
    pad_offset: isize
) -> T
    where T: CommonBounds + NormalOut<Output = T> + Mul<Output = T> + MulAdd<Output = T>
{
    let mut sum = T::NEG_INF;
    for _ in 0..outer {
        for i in 0..inner {
            let any = (0..kernel_ndim - 1).all(|x| {
                let a = reduce_prg[x] * dilations[x];
                a >= padding_bounds_left[x] && a < padding_bounds_right[x]
            });
            if
                any &&
                (i as i64) * dilations[kernel_ndim - 1] >= padding_bounds_left[kernel_ndim - 1] &&
                (i as i64) * dilations[kernel_ndim - 1] < padding_bounds_right[kernel_ndim - 1]
            {
                let val = inp_ptr[begin + i * cache - pad_offset];
                sum = val._add(sum);
            }
        }
        for j in (0..kernel_ndim - 1).rev() {
            if reduce_prg[j] < _kernel_shape[j] - 1 {
                reduce_prg[j] += 1;
                inp_ptr.offset(dilations[j] * inp_reduce_strides[j]);
                break;
            } else {
                reduce_prg[j] = 0;
                inp_ptr.offset(-dilations[j] * inp_reduce_strides[j] * (_kernel_shape[j] - 1));
            }
        }
    }
    sum
}
