use std::borrow::BorrowMut;

use crate::{tensor_base::_Tensor, Cpu, Tensor};
use hpt_common::{
    error::base::TensorError,
    shape::{shape::Shape, shape_utils::mt_intervals},
    Pointer,
};
use hpt_iterator::iterator_traits::ParStridedIteratorZip;
use hpt_iterator::TensorIterator;
use hpt_traits::{CommonBounds, TensorCreator, TensorInfo};
use hpt_types::dtype::TypeCommon;
use hpt_types::type_promote::NormalOutUnary;
use hpt_types::{
    into_scalar::Cast,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

impl<T, const DEVICE: usize> _Tensor<T, Cpu, DEVICE> {
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn layernorm<S>(
        &self,
        normalized_shape: S,
        gamma: Option<&_Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE>>,
        beta: Option<&_Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE>>,
        eps: T,
    ) -> std::result::Result<_Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE>, TensorError>
    where
        T: CommonBounds
            + Cast<<T as FloatOutBinary>::Output>
            + NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: CommonBounds
            + NormalOut<T, Output = <T as FloatOutBinary>::Output>
            + FloatOutBinary<Output = <T as FloatOutBinary>::Output>
            + FloatOutUnary<Output = <T as FloatOutBinary>::Output>,
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec:
            NormalOut<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>,
        S: Into<Shape>,
        usize: Cast<<T as FloatOutBinary>::Output>,
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
        let mut res = _Tensor::<<T as FloatOutBinary>::Output, Cpu, DEVICE>::empty(self.shape())?;
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
        let mut a_ptrs: Vec<Pointer<T>> = vec![];
        let mut res_ptrs: Vec<Pointer<<T as FloatOutBinary>::Output>> = vec![];
        let mut prgs = vec![];
        let mut task_amout = 0;
        let mut progress_init_a_data = vec![0; res_layout.ndim()];
        let a_ptr = self.ptr();
        let res_ptr = res.ptr();
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = a_ptr.clone();
            let mut res_data_ptr_cpy = res_ptr.clone();
            let a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();
            let res_data_ptr_cpy = res_data_ptr_cpy.borrow_mut();
            for i in (0..=res_layout.ndim() as i64 - 1).rev() {
                a_data_ptr_cpy
                    .offset(progress_init_a_data[i as usize] * self.strides()[i as usize]);
                res_data_ptr_cpy
                    .offset(progress_init_a_data[i as usize] * res.strides()[i as usize]);
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
                            let a_val = inp_ptr[i * inp_last_stride];
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
                    let mean = sum._div((inner_loop_size * inner_loop_size_2).cast());
                    prg.copy_from_slice(&prg_cpy);
                    let mut var = <T as FloatOutBinary>::Output::ZERO;
                    for _ in 0..inner_loop_size_2 {
                        for i in 0..inner_loop_size as i64 {
                            let a_val = inp_ptr[i * inp_last_stride];
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
                            let a_val = inp_ptr[i * inp_last_stride];
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
            (None, Some(beta)) => hpt_traits::NormalBinOps::add_(&res, beta, res.clone()),
            (Some(gamma), None) => hpt_traits::NormalBinOps::mul_(&res, gamma, res.clone()),
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
}

#[inline]

fn update_prg2<T>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut Pointer<T>,
    strides: &[i64],
    shape: &[i64],
) {
    for j in (0..shape_len - 1).rev() {
        let j = j as usize;

        if prg[j] < shape[j] {
            prg[j] += 1;
            inp_ptr.offset(strides[j]);
            break;
        } else {
            prg[j] = 0;
            inp_ptr.offset(-strides[j] * shape[j]);
        }
    }
}

#[inline]
fn update_prg3<T, O>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut Pointer<T>,
    res_ptr: &mut Pointer<O>,
    strides: &[i64],
    shape: &[i64],
    res_strides: &[i64],
    res_shape: &[i64],
) {
    for j in (0..shape_len - 1).rev() {
        let j = j as usize;

        if prg[j] < shape[j] {
            prg[j] += 1;
            inp_ptr.offset(strides[j]);
            res_ptr.offset(res_strides[j]);
            break;
        } else {
            prg[j] = 0;
            inp_ptr.offset(-strides[j] * shape[j]);
            res_ptr.offset(-res_strides[j] * res_shape[j]);
        }
    }
}

impl<T, const DEVICE: usize> Tensor<T, Cpu, DEVICE> {
    /// LayerNorm
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn layernorm<S>(
        &self,
        normalized_shape: S,
        gamma: Option<&Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE>>,
        beta: Option<&Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE>>,
        eps: T,
    ) -> Result<Tensor<<T as FloatOutBinary>::Output, Cpu, DEVICE>, TensorError>
    where
        T: CommonBounds
            + Cast<<T as FloatOutBinary>::Output>
            + NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: CommonBounds
            + NormalOut<T, Output = <T as FloatOutBinary>::Output>
            + FloatOutBinary<Output = <T as FloatOutBinary>::Output>
            + FloatOutUnary<Output = <T as FloatOutBinary>::Output>,
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec:
            NormalOut<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>,
        S: Into<Shape>,
        usize: Cast<<T as FloatOutBinary>::Output>,
    {
        Ok(self
            .inner
            .layernorm(
                normalized_shape,
                gamma.map(|gamma| gamma.inner.as_ref()),
                beta.map(|beta| beta.inner.as_ref()),
                eps,
            )?
            .into())
    }
}
