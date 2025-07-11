use crate::backend::Cpu;
use crate::ops::ShapeManipulate;
use crate::slice;
use crate::tensor_base::_Tensor;
use hpt_allocator::traits::Allocator;
use hpt_allocator::traits::AllocatorOutputRetrive;
use hpt_common::error::param::ParamError;
use hpt_common::error::shape::ShapeError;
use hpt_common::prg_update::next_sub1;
use hpt_common::shape::shape_utils::mt_intervals;
use hpt_common::{axis::axis::Axis, error::base::TensorError, shape::shape::Shape};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::shape_manipulate::Concat;
use hpt_traits::ops::slice::Slice;
use hpt_traits::ops::unary::Contiguous;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_traits::tensor::TensorLike;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::panic::Location;

impl<T: CommonBounds, const DEVICE: usize, Al> ShapeManipulate for _Tensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;
    type Output = _Tensor<T, Cpu, DEVICE, Al>;
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::squeeze(
            self,
            axes,
            |a| a.contiguous(),
        )?)
    }
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::unsqueeze(
            self,
            axes,
            |a| a.contiguous(),
        )?)
    }
    fn reshape<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::reshape(
            self,
            shape,
            |a| a.contiguous(),
        )?)
    }
    fn transpose(&self, axis1: i64, axis2: i64) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::transpose(
            self, axis1, axis2,
        )?)
    }
    #[track_caller]
    fn permute<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::permute(
            self,
            axes,
            |layout, axes| layout.permute(axes),
        )?)
    }

    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::permute(
            self,
            axes,
            |layout, axes| layout.permute_inv(axes),
        )?)
    }
    fn expand<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::expand(
            self, shape,
        )?)
    }
    fn t(&self) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::t(self)?)
    }
    fn mt(&self) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::mt(self)?)
    }
    fn flip<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::flip(self, axes)?)
    }
    fn fliplr(&self) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::fliplr(self)?)
    }
    fn flipud(&self) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::flipud(self)?)
    }
    fn tile<S: Into<Axis>>(&self, repeats: S) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::tile(
            self,
            repeats,
            |a| a.contiguous(),
        )?)
    }
    fn trim_zeros(&self, trim: &str) -> std::result::Result<Self, TensorError>
    where
        Self::Meta: PartialEq,
    {
        ParamError::check_trim(trim)?;
        if self.ndim() > 1 {
            return Err(ShapeError::InvalidDimension {
                message: "trim_zeros only support 1D tensor".to_string(),
                location: Location::caller(),
            }
            .into());
        }
        let stride = self.strides()[0] as isize;
        let raw = self.as_raw();
        let mut ptr = raw.as_ptr();
        let mut left_len = 0;
        if trim.contains('f') {
            unsafe {
                for i in 0..raw.len() as isize {
                    if *ptr.offset(i * stride) != T::ZERO {
                        break;
                    } else {
                        left_len += 1;
                    }
                }
            }
        }
        let mut right_len = raw.len() as i64;
        if trim.contains('b') {
            unsafe {
                ptr = raw.as_ptr().offset(((raw.len() - 1) as isize) * stride);
                let stride = -stride;
                for i in 0..raw.len() as isize {
                    if *ptr.offset(i * stride) != T::ZERO {
                        break;
                    } else {
                        right_len -= 1;
                    }
                }
            }
        }
        Ok(slice!(self[left_len:right_len])?)
    }
    fn repeat(&self, repeats: usize, axes: i16) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::repeat(
            self,
            repeats,
            axes,
            |a| a.contiguous(),
        )?)
    }
    fn split(
        &self,
        indices_or_sections: &[i64],
        axis: i64,
    ) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(crate::backends::common::shape_manipulate::split(
            self,
            indices_or_sections,
            axis,
        )?)
    }
    fn dsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(crate::backends::common::shape_manipulate::dsplit(
            self, indices,
        )?)
    }
    fn hsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(crate::backends::common::shape_manipulate::hsplit(
            self, indices,
        )?)
    }
    fn vsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(crate::backends::common::shape_manipulate::vsplit(
            self, indices,
        )?)
    }
    fn swap_axes(&self, axis1: i64, axis2: i64) -> std::result::Result<Self, TensorError> {
        Ok(crate::backends::common::shape_manipulate::swap_axes(
            self, axis1, axis2,
        )?)
    }
    fn flatten<A>(&self, start_dim: A, end_dim: A) -> std::result::Result<Self, TensorError>
    where
        A: Into<Option<usize>>,
    {
        Ok(crate::backends::common::shape_manipulate::flatten(
            self,
            start_dim,
            end_dim,
            |a| a.contiguous(),
        )?)
    }
}

impl<T: CommonBounds, const DEVICE: usize, Al> Concat for _Tensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator + Send + Sync + Clone + 'static,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cpu, DEVICE, Al>;
    fn concat(
        tensors: Vec<Self>,
        axis: usize,
        keepdims: bool,
    ) -> std::result::Result<Self, TensorError>
    where
        T: 'static,
    {
        let length = tensors.len();
        for i in tensors.iter() {
            for (idx, x) in tensors[0].shape().iter().enumerate() {
                if idx != axis
                    && i.shape().len() == tensors[0].shape().len()
                    && *x != i.shape()[idx]
                {
                    return Err(ShapeError::ConcatDimMismatch {
                        expected: *x as usize,
                        actual: i.shape()[idx] as usize,
                        location: Location::caller(),
                    }
                    .into());
                } else if i.shape().len() != tensors[0].shape().len() {
                    ShapeError::check_ndim_enough(
                        "concat dim mismatch".to_string(),
                        tensors[0].ndim(),
                        i.ndim(),
                    )?;
                }
            }
        }
        let mut new_shape: Vec<i64> = vec![0; tensors[0].ndim()];
        tensors.iter().for_each(|x| {
            new_shape[axis] += x.shape()[axis];
        });
        tensors[0].shape().iter().enumerate().for_each(|(i, x)| {
            if i != axis {
                new_shape[i] = *x;
            }
        });
        let new_tensor = Self::empty(&new_shape)?;
        let mut begin = 0;
        let mut res_slices = Vec::with_capacity(length);
        for i in tensors.iter() {
            let mut selections = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); new_shape.len()];
            selections[axis] = (begin, begin + i.shape()[axis], 1);
            begin += i.shape()[axis];
            let res_tensor = new_tensor.slice(&selections)?;
            res_slices.push(res_tensor);
        }
        let tensors = tensors
            .iter()
            .map(|x| (*x).clone())
            .collect::<Vec<_Tensor<T, Cpu, DEVICE, Al>>>();
        let num_threads = if length < rayon::current_num_threads() {
            length
        } else {
            rayon::current_num_threads()
        };
        let intervals = mt_intervals(length, num_threads);
        let res_tensors = intervals
            .iter()
            .map(|(start, end)| res_slices[*start..*end].to_vec())
            .collect::<Vec<_>>();
        let inputs = intervals
            .iter()
            .map(|(start, end)| tensors[*start..*end].to_vec())
            .collect::<Vec<_>>();
        res_tensors
            .into_par_iter()
            .zip(inputs.into_par_iter())
            .for_each(|(res_tensors, inputs)| {
                for (res, input) in res_tensors.into_iter().zip(inputs.into_iter()) {
                    let mut res_ptr = res.ptr::<T>();
                    let mut a_data = input.ptr::<T>();
                    let a_last_stride = *input.strides().last().unwrap();
                    let inner_loop_size = *input.shape().last().unwrap();
                    let outer_loop_size = input.size() / (inner_loop_size as usize);
                    let mut prg = vec![0; input.ndim()];
                    for _ in 0..outer_loop_size {
                        for i in 0..inner_loop_size {
                            res_ptr[i] = a_data[i * a_last_stride];
                        }
                        next_sub1(
                            &mut prg,
                            input.shape(),
                            [&mut a_data, &mut res_ptr],
                            [&input.shape(), &res.shape()],
                            [&input.strides(), &res.strides()],
                        );
                    }
                }
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
    fn vstack(tensors: Vec<Self>) -> std::result::Result<Self, TensorError> {
        Self::concat(tensors, 0, false)
    }
    fn hstack(mut tensors: Vec<Self>) -> std::result::Result<Self, TensorError> {
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 2 {
                return if tensor.shape().len() == 1 {
                    Self::concat(tensors, 0, false)
                } else {
                    // scalar
                    let mut tensors_ref = Vec::with_capacity(tensors.len());
                    let mut tensors_holder = Vec::with_capacity(tensors.len());
                    for tensor in tensors {
                        tensors_holder.push(tensor.reshape(vec![1])?);
                    }
                    for tensor in tensors_holder.into_iter() {
                        tensors_ref.push(tensor);
                    }
                    Self::concat(tensors_ref, 0, false)
                };
            }
        }
        Self::concat(tensors, 1, false)
    }
    fn dstack(mut tensors: Vec<Self>) -> std::result::Result<Self, TensorError> {
        let mut new_tensors = Vec::with_capacity(tensors.len());
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 3 {
                if tensor.shape().len() == 1 {
                    new_tensors.push(tensor.reshape(vec![1, tensor.shape()[0], 1])?);
                } else if tensor.shape().len() == 0 {
                    new_tensors.push(tensor.reshape(vec![1, 1, 1])?);
                } else {
                    new_tensors.push(tensor.reshape(vec![
                        tensor.shape()[0],
                        tensor.shape()[1],
                        1,
                    ])?);
                }
            } else {
                new_tensors.push(tensor.clone());
            }
        }
        let mut tensors_ref = Vec::with_capacity(new_tensors.len());
        for tensor in new_tensors.into_iter() {
            tensors_ref.push(tensor);
        }
        Self::concat(tensors_ref, 2, false)
    }
}
