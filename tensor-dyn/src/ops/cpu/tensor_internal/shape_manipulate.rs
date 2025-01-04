use crate::tensor_base::_Tensor;
use std::panic::Location;
use tensor_common::error::param::ParamError;
use tensor_common::error::shape::ShapeError;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_common::{axis::axis::Axis, error::base::TensorError, shape::shape::Shape};
use tensor_macros::match_selection;
use tensor_traits::{CommonBounds, ShapeManipulate, TensorInfo, TensorLike};

impl<T: CommonBounds> ShapeManipulate for _Tensor<T> {
    type Meta = T;
    type Output = _Tensor<T>;
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::squeeze(
            self,
            axes,
            |a| a.contiguous(),
        )?)
    }
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::unsqueeze(
            self,
            axes,
            |a| a.contiguous(),
        )?)
    }
    fn reshape<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::reshape(
            self,
            shape,
            |a| a.contiguous(),
        )?)
    }
    fn transpose(&self, axis1: i64, axis2: i64) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::transpose(
            self, axis1, axis2,
        )?)
    }
    fn permute<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::permute(
            self,
            axes,
            |layout, axes| layout.permute(axes),
        )?)
    }

    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::permute(
            self,
            axes,
            |layout, axes| layout.permute_inv(axes),
        )?)
    }
    fn expand<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::expand(self, shape)?)
    }
    fn t(&self) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::t(self)?)
    }
    fn mt(&self) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::mt(self)?)
    }
    fn flip<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::flip(self, axes)?)
    }
    fn fliplr(&self) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::fliplr(self)?)
    }
    fn flipud(&self) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::flipud(self)?)
    }
    fn tile<S: Into<Axis>>(&self, repeats: S) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::tile(
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
        Ok(crate::ops::common::shape_manipulate::repeat(
            self,
            repeats,
            axes,
            |a| a.contiguous(),
        )?)
    }
    fn split(&self, indices_or_sections: &[i64], axis: i64) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(crate::ops::common::shape_manipulate::split(
            self,
            indices_or_sections,
            axis,
        )?)
    }
    fn dsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(crate::ops::common::shape_manipulate::dsplit(self, indices)?)
    }
    fn hsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(crate::ops::common::shape_manipulate::hsplit(self, indices)?)
    }
    fn vsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(crate::ops::common::shape_manipulate::vsplit(self, indices)?)
    }
    fn swap_axes(&self, axis1: i64, axis2: i64) -> std::result::Result<Self, TensorError> {
        Ok(crate::ops::common::shape_manipulate::swap_axes(
            self, axis1, axis2,
        )?)
    }
    fn flatten<A>(&self, start_dim: A, end_dim: A) -> std::result::Result<Self, TensorError>
    where
        A: Into<Option<usize>>,
    {
        Ok(crate::ops::common::shape_manipulate::flatten(
            self,
            start_dim,
            end_dim,
            |a| a.contiguous(),
        )?)
    }
    fn concat(tensors: Vec<&_Tensor<T>>, axis: usize, keepdims: bool) -> std::result::Result<Self, TensorError>
    where
        T: 'static,
    {
        crate::ops::cpu::concat::concat(tensors, axis, keepdims)
    }
    fn vstack(tensors: Vec<&_Tensor<T>>) -> std::result::Result<Self, TensorError> {
        crate::ops::cpu::concat::concat(tensors, 0, false)
    }
    fn hstack(mut tensors: Vec<&_Tensor<T>>) -> std::result::Result<Self, TensorError> {
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 2 {
                return if tensor.shape().len() == 1 {
                    crate::ops::cpu::concat::concat(tensors, 0, false)
                } else {
                    // scalar
                    let mut tensors_ref = Vec::with_capacity(tensors.len());
                    let mut tensors_holder = Vec::with_capacity(tensors.len());
                    for tensor in tensors {
                        tensors_holder.push(tensor.reshape(vec![1])?);
                    }
                    for tensor in tensors_holder.iter() {
                        tensors_ref.push(tensor);
                    }
                    crate::ops::cpu::concat::concat(tensors_ref, 0, false)
                };
            }
        }
        crate::ops::cpu::concat::concat(tensors, 1, false)
    }
    fn dstack(mut tensors: Vec<&_Tensor<T>>) -> std::result::Result<Self, TensorError> {
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
        for tensor in new_tensors.iter() {
            tensors_ref.push(tensor);
        }
        crate::ops::cpu::concat::concat(tensors_ref, 2, false)
    }
}
