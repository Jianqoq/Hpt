use crate::tensor_base::_Tensor;
use crate::Cuda;
use anyhow::Result;
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

impl<T: CommonBounds, const DEVICE_ID: usize> ShapeManipulate for _Tensor<T, Cuda, DEVICE_ID> {
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
            todo!()
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
    fn tile<S: Into<Axis>>(&self, _repeats: S) -> Result<Self> {
        unimplemented!()
    }
    fn trim_zeros(&self, _trim: &str) -> Result<Self>
    where
        Self::Meta: PartialEq,
    {
        unimplemented!()
    }
    fn repeat(&self, _repeats: usize, _axes: i16) -> Result<Self> {
        unimplemented!()
    }
    fn split(&self, _indices_or_sections: &[i64], _axis: i64) -> Result<Vec<Self>> {
        unimplemented!()
    }
    fn dsplit(&self, _indices: &[i64]) -> Result<Vec<Self>> {
        unimplemented!()
    }
    fn hsplit(&self, _indices: &[i64]) -> Result<Vec<Self>> {
        unimplemented!()
    }
    fn vsplit(&self, _indices: &[i64]) -> Result<Vec<Self>> {
        unimplemented!()
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
    fn concat(_tensors: Vec<&Self>, _axis: usize, _keepdims: bool) -> Result<Self>
    where
        T: 'static,
    {
        unimplemented!()
    }
    fn vstack(_tensors: Vec<&Self>) -> Result<Self> {
        unimplemented!()
    }
    fn hstack(_tensors: Vec<&Self>) -> Result<Self> {
        unimplemented!()
    }
    fn dstack(_tensors: Vec<&Self>) -> Result<Self> {
        unimplemented!()
    }
}
