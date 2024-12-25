use crate::tensor_base::_Tensor;
use crate::Cuda;
use anyhow::Result;
use cudarc::driver::DeviceRepr;
use cudarc::types::CudaTypeName;
use std::panic::Location;
use tensor_common::shape_utils::{try_pad_shape, yield_one_after};
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_common::{
    axis::{process_axes, Axis},
    err_handler::ErrHandler,
    layout::Layout,
    shape::Shape,
    shape_utils::yield_one_before,
};
use tensor_macros::match_selection;
use tensor_traits::{CommonBounds, ShapeManipulate, TensorInfo, TensorLike};

impl<T: CommonBounds + DeviceRepr + CudaTypeName, const DEVICE_ID: usize> ShapeManipulate
    for _Tensor<T, Cuda, DEVICE_ID>
{
    type Meta = T;
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        for i in 0..axes.len() {
            if self.shape()[axes[i]] != 1 {
                return Err(ErrHandler::SqueezeError(
                    axes[i],
                    self.shape().clone(),
                    Location::caller(),
                )
                .into());
            }
        }
        let new_shape: Vec<i64> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| !axes.contains(&i))
            .map(|(_, &x)| x)
            .collect();
        self.reshape(new_shape)
    }
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        let mut res_shape: Vec<i64> = self.shape().to_vec();
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        axes.iter().for_each(|&x| {
            res_shape = yield_one_before(&res_shape, x);
        });
        self.reshape(res_shape)
    }
    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        if shape.size() != (self.size() as i64) {
            return Err(ErrHandler::ReshapeError(
                self.shape().clone(),
                shape.clone(),
                self.size(),
                shape.size() as usize,
                Location::caller(),
            )
            .into());
        }
        if let Ok(new_layout) = self.layout.inplace_reshape(&shape) {
            Ok(_Tensor {
                data: self.data.clone(),
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                layout: new_layout,
                _backend: self._backend.clone(),
            })
        } else {
            self.contiguous()?.reshape(shape)
        }
    }
    fn transpose(&self, axis1: i64, axis2: i64) -> Result<Self> {
        if self.ndim() < 2 {
            Err(
                ErrHandler::TransposeError(self.shape().clone(), self.ndim(), Location::caller())
                    .into(),
            )
        } else {
            self.permute(vec![axis1, axis2])
        }
    }
    fn permute<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        let permuted_layout = self.layout.permute(axes)?;
        Ok(_Tensor {
            data: self.data.clone(),
            layout: permuted_layout,
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            _backend: self._backend.clone(),
        })
    }

    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        let permuted_layout = self.layout.permute_inv(axes)?;
        Ok(_Tensor {
            data: self.data.clone(),
            layout: permuted_layout,
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            _backend: self._backend.clone(),
        })
    }
    fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let res_strides = self.layout.expand_strides(&res_shape);
        Ok(Self {
            data: self.data.clone(),
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            layout: Layout::new(res_shape, res_strides),
            _backend: self._backend.clone(),
        })
    }
    fn t(&self) -> Result<Self> {
        if self.ndim() > 2 {
            let mut axes = (0..self.ndim() as i64).collect::<Vec<i64>>();
            axes.swap(self.ndim() - 1, self.ndim() - 2);
            return self.permute(axes);
        }
        self.transpose(1, 0)
    }
    fn mt(&self) -> Result<Self> {
        self.permute((0..self.ndim() as i64).rev().collect::<Vec<i64>>())
    }
    fn flip<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        let axes = process_axes(axes, self.ndim())?;
        let mut new_strides = self.strides().to_vec();
        let mut ptr = self.ptr();
        for &i in axes.iter() {
            new_strides[i] = -new_strides[i];
            ptr.offset(self.strides()[i]);
        }
        if self.parent.is_none() {
            Ok(Self {
                data: ptr,
                parent: Some(self.data.clone()),
                mem_layout: self.mem_layout.clone(),
                layout: Layout::new(self.shape().clone(), new_strides),
                _backend: self._backend.clone(),
            })
        } else {
            Ok(Self {
                data: ptr,
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                layout: Layout::new(self.shape().clone(), new_strides),
                _backend: self._backend.clone(),
            })
        }
    }
    fn fliplr(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(ErrHandler::NdimNotEnough(2, self.ndim(), Location::caller()).into());
        }
        self.flip(1)
    }
    fn flipud(&self) -> Result<Self> {
        if self.ndim() < 1 {
            return Err(ErrHandler::NdimNotEnough(1, self.ndim(), Location::caller()).into());
        }
        self.flip(0)
    }
    fn tile<S: Into<Axis>>(&self, repeats: S) -> Result<Self> {
        let repeats: Axis = repeats.into();
        ErrHandler::check_index_in_range(self.ndim(), (repeats.axes.len() - 1) as i64)?;
        let repeats: Vec<i64> = repeats
            .axes
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>();
        let final_repeats;
        let mut final_shape;
        if repeats.len() > self.ndim() {
            final_shape = try_pad_shape(self.shape().as_ref(), repeats.len());
            final_repeats = repeats.clone();
        } else {
            final_shape = self.shape().to_vec();
            final_repeats = try_pad_shape(repeats.as_ref(), self.ndim());
        }
        let mut res = self.reshape(&final_shape)?;
        let mut cnt = 0;
        for (idx, &i) in final_repeats.iter().enumerate() {
            if i == 1 {
                continue;
            } else {
                let tmp_shape = yield_one_before(res.shape().as_ref(), idx);
                res = res.reshape(tmp_shape)?;
                res = res.repeat(i as usize, (idx + cnt) as i16)?;
                final_shape[idx] *= i;
                cnt += 1;
            }
        }
        res.reshape(final_shape)
    }
    fn trim_zeros(&self, trim: &str) -> Result<Self>
    where
        Self::Meta: PartialEq,
    {
        if !(trim == "fb" || trim == "f" || trim == "b") {
            return Err(anyhow::Error::msg("trim must be one of 'fb', 'f', 'b'"));
        }
        if self.ndim() > 1 {
            return Err(ErrHandler::NdimExceed(1, self.ndim(), Location::caller()).into());
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
    fn repeat(&self, repeats: usize, axes: i16) -> Result<Self> {
        let mut val: usize = axes as usize;
        if axes < 0 {
            val = self.shape().len() + (axes as usize);
        }
        let mut new_shape = yield_one_after(&self.shape(), val);
        let mut new_tensor: _Tensor<T, Cuda, DEVICE_ID> = self.reshape(&new_shape)?;
        new_shape[val + 1] *= repeats as i64;
        new_tensor = new_tensor.expand(new_shape)?;
        new_shape = self.shape().to_vec();
        new_shape[val] *= repeats as i64;
        Ok(new_tensor.contiguous()?.reshape(new_shape)?)
    }
    fn split(&self, indices_or_sections: &[i64], axis: i64) -> Result<Vec<Self>> {
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
    fn dsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 3 {
            return Err(
                ErrHandler::NdimNotEnough(3, self.shape().len(), Location::caller()).into(),
            );
        }
        self.split(indices, 2)
    }
    fn hsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 2 {
            return Err(
                ErrHandler::NdimNotEnough(2, self.shape().len(), Location::caller()).into(),
            );
        }
        self.split(indices, 1)
    }
    fn vsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 1 {
            return Err(
                ErrHandler::NdimNotEnough(1, self.shape().len(), Location::caller()).into(),
            );
        }
        self.split(indices, 0)
    }
    fn swap_axes(&self, mut axis1: i64, mut axis2: i64) -> Result<Self> {
        ErrHandler::check_index_in_range_mut(self.ndim(), &mut axis1)?;
        ErrHandler::check_index_in_range_mut(self.ndim(), &mut axis2)?;
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
    fn flatten<A>(&self, start_dim: A, end_dim: A) -> Result<Self>
    where
        A: Into<Option<usize>>,
    {
        let start = start_dim.into().unwrap_or(0);
        let end = end_dim.into().unwrap_or(self.ndim() - 1);
        let shape = self.shape();
        ErrHandler::check_index_in_range(self.ndim(), start as i64)?;
        ErrHandler::check_index_in_range(self.ndim(), end as i64)?;
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
    fn concat(tensors: Vec<&Self>, axis: usize, keepdims: bool) -> Result<Self>
    where
        T: 'static,
    {
        crate::ops::cuda::concat::concat(tensors, axis, keepdims)
    }
    fn vstack(tensors: Vec<&Self>) -> Result<Self> {
        crate::ops::cuda::concat::concat(tensors, 0, false)
    }
    fn hstack(mut tensors: Vec<&Self>) -> Result<Self> {
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
    fn dstack(mut tensors: Vec<&Self>) -> Result<Self> {
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
