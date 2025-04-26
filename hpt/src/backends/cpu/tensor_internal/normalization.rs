use std::borrow::BorrowMut;

use crate::backends::cpu::kernels::normalization::log_softmax::{
    contiguous_log_softmax, uncontiguous_log_softmax,
};
use crate::backends::cpu::kernels::normalization::softmax::{
    contiguous_softmax, uncontiguous_softmax,
};
use crate::iter::TensorIterator;
use crate::tensor_base::_Tensor;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::shape::shape::Shape;
use hpt_common::shape::shape_utils::mt_intervals;
use hpt_common::Pointer;
use hpt_iterator::iterator_traits::ParStridedIteratorZip;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::normalization::NormalizationOps;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::into_vec::IntoVec;
use hpt_types::type_promote::{FloatOutBinary, FloatOutUnary, NormalOut, NormalOutUnary};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE: usize, A> NormalizationOps for _Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds
        + FloatOutBinary
        + Cast<FloatBinaryType<T>>
        + FloatOutUnary<Output = FloatBinaryType<T>>,
    T::Vec: FloatOutUnary<Output = <FloatBinaryType<T> as TypeCommon>::Vec>
        + IntoVec<<FloatBinaryType<T> as TypeCommon>::Vec>,
    FloatBinaryType<T>: CommonBounds
        + FloatOutUnary<Output = FloatBinaryType<T>>
        + NormalOut<T, Output = FloatBinaryType<T>>,
    <FloatBinaryType<T> as TypeCommon>::Vec:
        FloatOutUnary<Output = <FloatBinaryType<T> as TypeCommon>::Vec>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<FloatBinaryType<T>, Cpu, DEVICE, A>;

    type OutputMeta = FloatBinaryType<T>;

    fn layernorm<S: Into<Shape>>(
        &self,
        normalized_shape: S,
        gamma: Option<&Self::Output>,
        beta: Option<&Self::Output>,
        eps: Self::OutputMeta,
    ) -> Result<Self::Output, TensorError>
    where
        usize: Cast<Self::OutputMeta>,
    {
        let normalized_shape: Shape = normalized_shape.into();
        let mut axes = Vec::new();
        for ((i, &dim), &ns) in self
            .shape()
            .iter()
            .enumerate()
            .rev()
            .zip(normalized_shape.iter().rev())
        {
            if dim != ns {
                panic!("normalized dims must match last dims of input tensor, shape: {}, normalized_shape: {:?}", self.shape(), normalized_shape);
            }
            axes.push(i);
        }
        let mut res = Self::Output::empty(self.shape())?;
        let res_layout = self.layout.reduce(axes, false)?;
        let inner_loop_size = *self.shape().last().unwrap() as usize;
        let outer_loop_size = self.size() / inner_loop_size;
        let inner_loop_size_2 = outer_loop_size / res_layout.size() as usize;

        let num_threads = if (res_layout.size() as usize) < rayon::current_num_threads() {
            res_layout.size() as usize
        } else {
            rayon::current_num_threads()
        };
        let intervals = mt_intervals(res_layout.size() as usize, num_threads);
        let mut a_ptrs = vec![];
        let mut res_ptrs = vec![];
        let mut prgs = vec![];
        let mut task_amout = 0;
        let mut progress_init_a_data = vec![0; res_layout.ndim()];
        let a_ptr = self.ptr();
        let res_ptr = res.ptr();
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = a_ptr.clone();
            let mut res_data_ptr_cpy = res_ptr.clone();
            let mut a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();
            let mut res_data_ptr_cpy = res_data_ptr_cpy.borrow_mut();
            for i in (0..=res_layout.ndim() as i64 - 1).rev() {
                a_data_ptr_cpy
                    += progress_init_a_data[i as usize] * self.strides()[i as usize];
                res_data_ptr_cpy
                    += progress_init_a_data[i as usize] * res.strides()[i as usize];
            }
            let mut tmp1 = (task_amout * inner_loop_size) as i64;
            let mut prg = vec![0; self.ndim() - 1];
            for i in (0..=self.ndim() as i64 - 2).rev() {
                prg[i as usize] = tmp1 % self.shape()[i as usize];
                tmp1 /= self.shape()[i as usize];
            }
            task_amout += intervals[id].1 - intervals[id].0;
            let mut tmp2 = task_amout as i64;
            for j in (0..=res_layout.ndim() as i64 - 1).rev() {
                progress_init_a_data[j as usize] = tmp2 % res_layout.shape()[j as usize];
                tmp2 /= res_layout.shape()[j as usize];
            }
            a_ptrs.push(a_data_ptr_cpy.clone());
            res_ptrs.push(res_data_ptr_cpy.clone());
            prgs.push(prg);
        }

        let inp_last_stride = *self.strides().last().unwrap();
        intervals
            .into_par_iter()
            .zip(a_ptrs.into_par_iter())
            .zip(res_ptrs.into_par_iter())
            .zip(prgs.into_par_iter())
            .for_each(|(((interval, mut inp_ptr), mut res_ptr), mut prg)| {
                for _ in 0..interval.1 - interval.0 {
                    let mut sum = <T as FloatOutBinary>::Output::ZERO;
                    let prg_cpy = prg.clone();
                    let inp_ptr_origin = inp_ptr.clone();
                    for _ in 0..inner_loop_size_2 {
                        for i in 0..inner_loop_size as i64 {
                            let a_val: Self::OutputMeta = inp_ptr[i * inp_last_stride].cast();
                            sum = sum._add(a_val);
                        }
                        update_prg2(
                            &mut prg,
                            self.ndim() as i64,
                            &mut inp_ptr,
                            self.strides(),
                            self.shape(),
                        );
                    }
                    inp_ptr = inp_ptr_origin.clone();
                    let s: Self::OutputMeta = (inner_loop_size * inner_loop_size_2).cast();
                    let mean: Self::OutputMeta = sum._div(s);
                    prg.copy_from_slice(&prg_cpy);
                    let mut var = <T as FloatOutBinary>::Output::ZERO;
                    for _ in 0..inner_loop_size_2 {
                        for i in 0..inner_loop_size as i64 {
                            let a_val: FloatBinaryType<T> = inp_ptr[i * inp_last_stride].cast();
                            let sub = a_val._sub(mean)._square();
                            var = var._add(sub);
                        }
                        update_prg2(
                            &mut prg,
                            self.ndim() as i64,
                            &mut inp_ptr,
                            self.strides(),
                            self.shape(),
                        );
                    }
                    inp_ptr = inp_ptr_origin.clone();
                    prg.copy_from_slice(&prg_cpy);
                    var = var._div((inner_loop_size * inner_loop_size_2).cast());
                    for _ in 0..inner_loop_size_2 {
                        for i in 0..inner_loop_size as i64 {
                            let a_val: FloatBinaryType<T> = inp_ptr[i * inp_last_stride].cast();
                            let sub = a_val._sub(mean)._div(var._add(eps)._sqrt());
                            res_ptr[i] = sub;
                        }
                        update_prg3(
                            &mut prg,
                            self.ndim() as i64,
                            &mut inp_ptr,
                            &mut res_ptr,
                            self.strides(),
                            self.shape(),
                            res.strides(),
                            res.shape(),
                        );
                    }
                }
            });
        match (gamma, beta) {
            (None, None) => Ok(res),
            (None, Some(beta)) => {
                hpt_traits::ops::binary::NormalBinOps::add_(&res, beta, res.clone())
            }
            (Some(gamma), None) => {
                hpt_traits::ops::binary::NormalBinOps::mul_(&res, gamma, res.clone())
            }
            (Some(gamma), Some(beta)) => {
                res.par_iter_mut()
                    .zip(gamma.par_iter())
                    .zip(beta.par_iter())
                    .for_each(|((res, gamma), beta)| {
                        *res = gamma._mul(*res)._add(beta);
                    });
                Ok(res)
            }
        }
    }

    fn softmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        let res = if self.is_contiguous() && self.parent().is_none() {
            contiguous_softmax(self, axis, None::<Self::Output>)?
        } else {
            uncontiguous_softmax(self, axis, None::<Self::Output>)?
        };
        Ok(res)
    }

    fn log_softmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        let res = if self.is_contiguous() && self.parent().is_none() {
            contiguous_log_softmax(self, axis, None::<Self::Output>)?
        } else {
            uncontiguous_log_softmax(self, axis, None::<Self::Output>)?
        };
        Ok(res)
    }
}

#[inline]
fn update_prg2<T>(
    prg: &mut [i64],
    shape_len: i64,
    mut inp_ptr: &mut Pointer<T>,
    strides: &[i64],
    shape: &[i64],
) {
    for j in (0..shape_len - 1).rev() {
        let j = j as usize;

        if prg[j] < shape[j] {
            prg[j] += 1;
            inp_ptr += strides[j];
            break;
        } else {
            prg[j] = 0;
            inp_ptr += -strides[j] * shape[j];
        }
    }
}

#[inline]
fn update_prg3<T, O>(
    prg: &mut [i64],
    shape_len: i64,
    mut inp_ptr: &mut Pointer<T>,
    mut res_ptr: &mut Pointer<O>,
    strides: &[i64],
    shape: &[i64],
    res_strides: &[i64],
    res_shape: &[i64],
) {
    for j in (0..shape_len - 1).rev() {
        let j = j as usize;

        if prg[j] < shape[j] {
            prg[j] += 1;
            inp_ptr += strides[j];
            res_ptr += res_strides[j];
            break;
        } else {
            prg[j] = 0;
            inp_ptr += -strides[j] * shape[j];
            res_ptr += -res_strides[j] * res_shape[j];
        }
    }
}
