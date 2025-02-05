use crate::tensor_base::_Tensor;
use crate::Cuda;
use cudarc::driver::DeviceRepr;
use cudarc::types::CudaTypeName;
use std::panic::Location;
use tensor_common::shape::shape_utils::yield_one_after;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_common::{axis::Axis, err_handler::TensorError, layout::Layout, shape::Shape};
use tensor_macros::match_selection;
use tensor_traits::{CommonBounds, ShapeManipulate, TensorInfo, TensorLike};

impl<T: CommonBounds + DeviceRepr + CudaTypeName, const DEVICE_ID: usize> ShapeManipulate
    for _Tensor<T, Cuda, DEVICE_ID>
{
    type Meta = T;
    type Output = Self;
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
        if !(trim == "fb" || trim == "f" || trim == "b") {
            return Err(TensorError::TrimError(trim.to_string(), Location::caller()).into());
        }
        if self.ndim() > 1 {
            return Err(TensorError::NdimExceed(1, self.ndim(), Location::caller()).into());
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
        slice!(self[left_len:right_len])
    }
    fn repeat(&self, repeats: usize, axes: i16) -> std::result::Result<Self, TensorError> {
        let mut val: usize = axes as usize;
        if axes < 0 {
            val = self.shape().len() + (axes as usize);
        }
        let mut new_shape = yield_one_after(&self.shape(), val);
        let mut new_tensor: Self = self.reshape(&new_shape)?;
        new_shape[val + 1] *= repeats as i64;
        new_tensor = new_tensor.expand(new_shape)?;
        new_shape = self.shape().to_vec();
        new_shape[val] *= repeats as i64;
        Ok(new_tensor.contiguous()?.reshape(new_shape)?)
    }
    fn split(
        &self,
        indices_or_sections: &[i64],
        axis: i64,
    ) -> std::result::Result<Vec<Self>, TensorError> {
        let mut new_axis = axis;
        if axis < 0 {
            new_axis = (self.ndim() as i64) + axis;
        }
        assert!(new_axis >= 0);
        let mut reses = vec![];
        let mut tmp: Vec<Slice> = Vec::with_capacity(self.ndim());
        for _ in 0..self.ndim() {
            tmp.push(Slice::Full);
        }
        let mut prev = 0;
        for &i in indices_or_sections.iter() {
            tmp[axis as usize] = Slice::Range((prev, i));
            prev = i;
            reses.push(self.slice(&tmp)?);
        }
        let last = *indices_or_sections.last().unwrap();
        tmp[axis as usize] = Slice::Range((last, self.shape()[axis as usize]));
        let remain = self.slice(&tmp)?;
        reses.push(remain);
        Ok(reses)
    }
    fn dsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        if self.shape().len() < 3 {
            return Err(
                TensorError::NdimNotEnough(3, self.shape().len(), Location::caller()).into(),
            );
        }
        self.split(indices, 2)
    }
    fn hsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        if self.shape().len() < 2 {
            return Err(
                TensorError::NdimNotEnough(2, self.shape().len(), Location::caller()).into(),
            );
        }
        self.split(indices, 1)
    }
    fn vsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        if self.shape().len() < 1 {
            return Err(
                TensorError::NdimNotEnough(1, self.shape().len(), Location::caller()).into(),
            );
        }
        self.split(indices, 0)
    }
    fn swap_axes(&self, mut axis1: i64, mut axis2: i64) -> std::result::Result<Self, TensorError> {
        TensorError::check_index_in_range_mut(self.ndim(), &mut axis1)?;
        TensorError::check_index_in_range_mut(self.ndim(), &mut axis2)?;
        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();
        new_shape.swap(axis1 as usize, axis2 as usize);
        new_strides.swap(axis1 as usize, axis2 as usize);
        let layout = Layout::new(new_shape, new_strides);
        Ok(Self {
            data: self.data.clone(),
            layout,
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            _backend: self._backend.clone(),
        })
    }
    fn flatten<A>(&self, start_dim: A, end_dim: A) -> std::result::Result<Self, TensorError>
    where
        A: Into<Option<usize>>,
    {
        let start = start_dim.into().unwrap_or(0);
        let end = end_dim.into().unwrap_or(self.ndim() - 1);
        let shape = self.shape();
        TensorError::check_index_in_range(self.ndim(), start as i64)?;
        TensorError::check_index_in_range(self.ndim(), end as i64)?;
        let flattened_dim = shape[start..=end].iter().product::<i64>();
        let mut new_shape = Vec::new();
        for (i, &dim) in shape.iter().enumerate() {
            if i < start {
                new_shape.push(dim);
            } else if i == start {
                new_shape.push(flattened_dim);
            } else if i > end {
                new_shape.push(dim);
            }
        }
        self.reshape(new_shape)
    }
    fn concat(
        tensors: Vec<&Self>,
        axis: usize,
        keepdims: bool,
    ) -> std::result::Result<Self, TensorError>
    where
        T: 'static,
    {
        crate::ops::cuda::concat::concat(tensors, axis, keepdims)
    }
    fn vstack(tensors: Vec<&Self>) -> std::result::Result<Self, TensorError> {
        crate::ops::cuda::concat::concat(tensors, 0, false)
    }
    fn hstack(mut tensors: Vec<&Self>) -> std::result::Result<Self, TensorError> {
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 2 {
                return if tensor.shape().len() == 1 {
                    crate::ops::cuda::concat::concat(tensors, 0, false)
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
                    crate::ops::cuda::concat::concat(tensors_ref, 0, false)
                };
            }
        }
        crate::ops::cuda::concat::concat(tensors, 1, false)
    }
    fn dstack(mut tensors: Vec<&Self>) -> std::result::Result<Self, TensorError> {
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
        crate::ops::cuda::concat::concat(tensors_ref, 2, false)
    }
}
