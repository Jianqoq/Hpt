use std::panic::Location;

use hpt_common::{
    Pointer,
    axis::axis::{Axis, process_axes},
    error::{base::TensorError, shape::ShapeError},
    layout::layout::Layout,
    shape::{
        shape::Shape,
        shape_utils::{try_pad_shape, yield_one_after, yield_one_before},
    },
    slice::slice_process,
};

use crate::{
    Tensor,
    utils::index_cal::{dispatch_loop_progress_update, dispatch_map_global_idx, dispatch_map_gp},
};

impl Tensor {
    pub fn reshape(&self, shape: &[i64]) -> Result<Tensor, TensorError> {
        let shape: Shape = shape.into();
        ShapeError::check_size_match(self.layout.size() as i64, shape.size())?;
        if let Ok(new_layout) = self.layout.inplace_reshape(&shape) {
            let prg_update = dispatch_loop_progress_update(&new_layout, self.dtype.sizeof());
            let map_global_idx = dispatch_map_global_idx(&new_layout);
            let map_gp = dispatch_map_gp(&new_layout);
            Ok(Tensor {
                data: self.data.clone(),
                layout: new_layout,
                dtype: self.dtype.clone(),
                device: self.device.clone(),
                parent: self.parent.clone(),
                prg_update,
                map_global_idx,
                map_gp,
                mem_layout: self.mem_layout.clone(),
                backend: self.backend.clone(),
            })
        } else {
            let a = self.contiguous()?;
            a.reshape(&shape)
        }
    }
    pub fn squeeze(&self, axes: &[i64]) -> Result<Tensor, TensorError> {
        let axes: Vec<usize> = process_axes(axes, self.layout.ndim())?;
        for i in 0..axes.len() {
            if self.layout.shape()[axes[i]] != 1 {
                return Err(ShapeError::SqueezeError {
                    axis: axes[i],
                    shape: self.layout.shape().clone(),
                    location: Location::caller(),
                }
                .into());
            }
        }
        let new_shape: Vec<i64> = self
            .layout
            .shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| !axes.contains(&i))
            .map(|(_, &x)| x)
            .collect();
        self.reshape(&new_shape)
    }
    pub fn unsqueeze(&self, axes: &[i64]) -> Result<Tensor, TensorError> {
        let mut res_shape: Vec<i64> = self.layout.shape().to_vec();
        let axes: Axis = axes.into();
        assert_eq!(axes.axes.len(), 1);
        let mut new_axes = Vec::with_capacity(axes.axes.len());
        for i in &axes.axes {
            if *i < 0 {
                let new_i = *i + self.layout.ndim() as i64;
                if new_i < 0 || new_i > self.layout.ndim() as i64 {
                    return Err(ShapeError::DimOutOfRange {
                        expected: 0..self.layout.ndim() as i64 + 1,
                        actual: new_i,
                        location: core::panic::Location::caller(),
                    }
                    .into());
                } else {
                    new_axes.push(new_i as usize);
                }
            } else {
                if *i > self.layout.ndim() as i64 {
                    return Err(ShapeError::DimOutOfRange {
                        expected: 0..self.layout.ndim() as i64 + 1,
                        actual: *i,
                        location: core::panic::Location::caller(),
                    }
                    .into());
                } else {
                    new_axes.push(*i as usize);
                }
            }
        }
        new_axes.iter().for_each(|&x| {
            res_shape = yield_one_before(&res_shape, x);
        });
        self.reshape(&res_shape)
    }

    pub fn permute(&self, axes: &[i64]) -> Result<Tensor, TensorError> {
        let permuted_layout = self.layout.permute(axes)?;
        let prg_update = dispatch_loop_progress_update(&permuted_layout, self.dtype.sizeof());
        let map_global_idx = dispatch_map_global_idx(&permuted_layout);
        let map_gp = dispatch_map_gp(&permuted_layout);
        Ok(Tensor {
            data: self.data.clone(),
            layout: permuted_layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
            prg_update,
            map_global_idx,
            map_gp,
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
        })
    }

    pub fn permute_inv(&self, axes: &[i64]) -> Result<Tensor, TensorError> {
        let permuted_layout = self.layout.permute_inv(axes)?;
        let prg_update = dispatch_loop_progress_update(&permuted_layout, self.dtype.sizeof());
        let map_global_idx = dispatch_map_global_idx(&permuted_layout);
        let map_gp = dispatch_map_gp(&permuted_layout);
        Ok(Tensor {
            data: self.data.clone(),
            layout: permuted_layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
            prg_update,
            map_global_idx,
            map_gp,
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
        })
    }

    pub fn expand(&self, shape: &[i64]) -> Result<Tensor, TensorError> {
        let res_shape = Shape::from(shape);
        let res_strides = self.layout.expand_strides(&res_shape)?;
        let res_layout = Layout::new(res_shape, res_strides);
        let prg_update = dispatch_loop_progress_update(&res_layout, self.dtype.sizeof());
        let map_global_idx = dispatch_map_global_idx(&res_layout);
        let map_gp = dispatch_map_gp(&res_layout);
        Ok(Tensor {
            data: self.data.clone(),
            layout: res_layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
            prg_update,
            map_global_idx,
            map_gp,
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
        })
    }

    pub fn transpose(&self, axis1: i64, axis2: i64) -> Result<Tensor, TensorError> {
        ShapeError::check_ndim_enough(
            "transpose expected 2 dimensions.".to_string(),
            2,
            self.layout.ndim(),
        )?;
        self.permute(&[axis1, axis2])
    }

    pub fn t(&self) -> Result<Tensor, TensorError> {
        if self.layout.ndim() > 2 {
            let mut axes = (0..self.layout.ndim() as i64).collect::<Vec<i64>>();
            axes.swap(self.layout.ndim() - 1, self.layout.ndim() - 2);
            return self.permute(&axes);
        }
        self.transpose(1, 0)
    }

    pub fn mt(&self) -> Result<Tensor, TensorError> {
        let axes = (0..self.layout.ndim() as i64).rev().collect::<Vec<i64>>();
        self.permute(&axes)
    }

    pub fn flip(&self, axes: &[i64]) -> Result<Tensor, TensorError> {
        let axes = process_axes(axes, self.layout.ndim())?;
        let mut new_strides = self.layout.strides().to_vec();
        let mut ptr = self.data.clone();
        let sizeof = self.dtype.sizeof() as i64;
        for &i in axes.iter() {
            ptr += new_strides[i] * (self.layout.shape()[i] - 1) * sizeof;
            new_strides[i] = -new_strides[i];
        }
        let new_parent = if self.parent.is_none() {
            Some(self.data.clone())
        } else {
            self.parent.clone()
        };
        let new_layout = Layout::new(self.layout.shape().clone(), new_strides);
        let prg_update = dispatch_loop_progress_update(&new_layout, self.dtype.sizeof());
        let map_global_idx = dispatch_map_global_idx(&new_layout);
        let map_gp = dispatch_map_gp(&new_layout);
        Ok(Tensor {
            data: ptr,
            parent: new_parent,
            mem_layout: self.mem_layout.clone(),
            layout: new_layout,
            backend: self.backend.clone(),
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            prg_update,
            map_global_idx,
            map_gp,
        })
    }

    pub fn fliplr(&self) -> Result<Tensor, TensorError> {
        ShapeError::check_ndim_enough(
            "fliplr expected 2 dimensions.".to_string(),
            2,
            self.layout.ndim(),
        )?;
        self.flip(&[1])
    }

    pub fn flipud(&self) -> Result<Tensor, TensorError> {
        ShapeError::check_ndim_enough(
            "flipud expected 2 dimensions.".to_string(),
            2,
            self.layout.ndim(),
        )?;
        self.flip(&[0])
    }

    pub fn tile(&self, repeats: &[i64]) -> Result<Tensor, TensorError> {
        let repeats: Axis = repeats.into();
        ShapeError::check_index_out_of_range(
            (repeats.axes.len() - 1) as i64,
            self.layout.ndim() as i64,
        )?;
        let repeats: Vec<i64> = repeats
            .axes
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>();
        let final_repeats;
        let mut final_shape;
        if repeats.len() > self.layout.ndim() {
            final_shape = try_pad_shape(self.layout.shape().as_ref(), repeats.len());
            final_repeats = repeats.clone();
        } else {
            final_shape = self.layout.shape().to_vec();
            final_repeats = try_pad_shape(repeats.as_ref(), self.layout.ndim());
        }
        let mut res = self.reshape(&final_shape)?;
        let mut cnt = 0;
        for (idx, &i) in final_repeats.iter().enumerate() {
            if i == 1 {
                continue;
            } else {
                let tmp_shape = yield_one_before(res.layout.shape().as_ref(), idx);
                res = res.reshape(&tmp_shape)?;
                res = res.repeat(i as usize, (idx + cnt) as i64)?;
                final_shape[idx] *= i;
                cnt += 1;
            }
        }
        res.reshape(&final_shape)
    }

    pub fn repeat(&self, repeats: usize, axes: i64) -> Result<Tensor, TensorError> {
        let mut val: usize = axes as usize;
        if axes < 0 {
            val = self.layout.ndim() + (axes as usize);
        }
        let mut new_shape = yield_one_after(&self.layout.shape(), val);
        let mut new_tensor = self.reshape(&new_shape)?;
        new_shape[val + 1] *= repeats as i64;
        new_tensor = new_tensor.expand(&new_shape)?;
        new_shape = self.layout.shape().to_vec();
        new_shape[val] *= repeats as i64;
        Ok(new_tensor.reshape(&new_shape)?)
    }

    pub fn split(
        &self,
        indices_or_sections: &[i64],
        axis: i64,
    ) -> Result<Vec<Tensor>, TensorError> {
        let mut new_axis = axis;
        if axis < 0 {
            new_axis = (self.layout.ndim() as i64) + axis;
        }
        assert!(new_axis >= 0);
        let mut reses = vec![];
        let mut tmp: Vec<(i64, i64, i64)> = Vec::with_capacity(self.layout.ndim());
        for _ in 0..self.layout.ndim() {
            tmp.push((0, 0x7FFFFFFFFFFFFFFF, 1));
        }
        let mut prev = 0;
        for &i in indices_or_sections.iter() {
            tmp[axis as usize] = (prev, i, 1);
            prev = i;
            reses.push(self.slice(&tmp)?);
        }
        let last = *indices_or_sections.last().unwrap();
        tmp[axis as usize] = (last, self.layout.shape()[axis as usize], 1);
        let remain = self.slice(&tmp)?;
        reses.push(remain);
        Ok(reses)
    }

    pub fn dsplit(&self, indices: &[i64]) -> Result<Vec<Tensor>, TensorError> {
        ShapeError::check_ndim_enough(
            "dsplit required input has 3 dimensions.".to_string(),
            3,
            self.layout.ndim(),
        )?;
        self.split(indices, 2)
    }

    pub fn hsplit(&self, indices: &[i64]) -> Result<Vec<Tensor>, TensorError> {
        ShapeError::check_ndim_enough(
            "hsplit required input has 2 dimensions.".to_string(),
            2,
            self.layout.ndim(),
        )?;
        self.split(indices, 1)
    }

    pub fn vsplit(&self, indices: &[i64]) -> Result<Vec<Tensor>, TensorError> {
        ShapeError::check_ndim_enough(
            "vsplit required input has 1 dimension.".to_string(),
            1,
            self.layout.ndim(),
        )?;
        self.split(indices, 0)
    }

    pub fn swapaxes(&self, mut axis1: i64, mut axis2: i64) -> Result<Tensor, TensorError> {
        if axis1 < 0 {
            axis1 += self.layout.ndim() as i64;
        }
        if axis2 < 0 {
            axis2 += self.layout.ndim() as i64;
        }
        ShapeError::check_index_out_of_range(axis1, self.layout.ndim() as i64)?;
        ShapeError::check_index_out_of_range(axis2, self.layout.ndim() as i64)?;
        let mut new_shape = self.layout.shape().to_vec();
        let mut new_strides = self.layout.strides().to_vec();
        new_shape.swap(axis1 as usize, axis2 as usize);
        new_strides.swap(axis1 as usize, axis2 as usize);
        let layout = Layout::new(new_shape, new_strides);
        let prg_update = dispatch_loop_progress_update(&layout, self.dtype.sizeof());
        let map_global_idx = dispatch_map_global_idx(&layout);
        let map_gp = dispatch_map_gp(&layout);
        Ok(Tensor {
            data: self.data.clone(),
            layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
            prg_update,
            map_global_idx,
            map_gp,
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
        })
    }

    pub fn flatten(&self, start_dim: i64, end_dim: i64) -> Result<Tensor, TensorError> {
        let start = start_dim as usize;
        let end = end_dim as usize;
        let shape = self.layout.shape();
        ShapeError::check_index_out_of_range(start as i64, self.layout.ndim() as i64)?;
        ShapeError::check_index_out_of_range(end as i64, self.layout.ndim() as i64)?;
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
        self.reshape(&new_shape)
    }

    pub fn broadcast_to(&self, shape: &[i64]) -> Result<Tensor, TensorError> {
        let res_shape = Shape::from(shape);
        let broadcasted_layout = self.layout.broadcast(&res_shape)?;
        let prg_update = dispatch_loop_progress_update(&broadcasted_layout, self.dtype.sizeof());
        let map_global_idx = dispatch_map_global_idx(&broadcasted_layout);
        let map_gp = dispatch_map_gp(&broadcasted_layout);
        Ok(Tensor {
            data: self.data.clone(),
            layout: broadcasted_layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
            prg_update,
            map_global_idx,
            map_gp,
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
        })
    }

    pub fn slice(&self, index: &[(i64, i64, i64)]) -> Result<Tensor, TensorError> {
        let (res_shape, res_strides, offset) = slice_process(
            self.layout.shape().to_vec(),
            self.layout.strides().to_vec(),
            index,
            1,
        )?;
        let res_ptr = unsafe {
            self.data
                .ptr
                .offset(offset as isize * self.dtype.sizeof() as isize)
        };
        #[cfg(feature = "bound_check")]
        {
            if offset < 0 || offset >= (self.data.len as i64) {
                panic!(
                    "index out of bounds, got offset: {}, origin shape: {}, origin strides: {}, slices: {:?}",
                    offset,
                    self.layout.shape(),
                    self.layout.strides(),
                    index
                );
            }
            let len = self.data.len - offset;
            Ok(from_slice(
                self,
                Pointer::new(res_ptr, len),
                res_shape,
                res_strides,
            ))
        }
        #[cfg(not(feature = "bound_check"))]
        {
            let layout = Layout::new(res_shape, res_strides);
            let new_parent = if self.parent.is_none() {
                Some(self.data.clone())
            } else {
                self.parent.clone()
            };
            let prg_update = dispatch_loop_progress_update(&layout, self.dtype.sizeof());
            let map_global_idx = dispatch_map_global_idx(&layout);
            let map_gp = dispatch_map_gp(&layout);
            Ok(Tensor {
                data: Pointer::new(res_ptr, 0),
                layout,
                dtype: self.dtype.clone(),
                device: self.device.clone(),
                parent: new_parent,
                prg_update,
                map_global_idx,
                map_gp,
                mem_layout: self.mem_layout.clone(),
                backend: self.backend.clone(),
            })
        }
    }
}
