use std::panic::Location;

use hpt_common::{
    Pointer,
    axis::axis::{Axis, process_axes},
    error::{base::TensorError, shape::ShapeError},
    layout::layout::Layout,
    shape::{
        shape::Shape,
        shape_utils::{mt_intervals, try_pad_shape, yield_one_after, yield_one_before},
    },
    slice::slice_process,
};
use hpt_traits::tensor::TensorInfo;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{
    Tensor, current_num_threads,
};

impl Tensor {
    pub fn reshape(&self, shape: &[i64]) -> Result<Tensor, TensorError> {
        let mut new_shape = shape.to_vec();

        let total_elements = self.size() as i64;

        let mut negative_one_index = None;
        let mut known_dims_product = 1;

        for (i, &dim) in shape.iter().enumerate() {
            if dim == -1 {
                if negative_one_index.is_some() {
                    panic!("there can only be one -1 dimension");
                }
                negative_one_index = Some(i);
            } else if dim <= 0 && dim != -1 {
                panic!("dimension size must be positive or -1");
            } else {
                known_dims_product *= dim;
            }
        }
        if let Some(idx) = negative_one_index {
            if known_dims_product == 0 {
                panic!(
                    "cannot infer the size of the -1 dimension because the other dimensions contain zero"
                );
            }

            if total_elements % known_dims_product != 0 {
                panic!(
                    "the total number of elements {} cannot be divided by the product of the known dimensions {}",
                    total_elements, known_dims_product
                );
            }

            new_shape[idx] = total_elements / known_dims_product;
        } else {
            if known_dims_product != total_elements {
                panic!(
                    "the number of elements after reshape {} is different from the original number of elements {}",
                    known_dims_product, total_elements
                );
            }
        }

        let shape: Shape = new_shape.as_slice().into();
        ShapeError::check_size_match(self.layout.size() as i64, shape.size())?;
        if let Ok(new_layout) = self.layout.inplace_reshape(&shape) {
            Ok(Tensor {
                data: self.data.clone(),
                layout: new_layout,
                dtype: self.dtype.clone(),
                device: self.device.clone(),
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                backend: self.backend.clone(),
            })
        } else {
            let a = self.contiguous()?;
            a.reshape(&shape)
        }
    }

    #[track_caller]
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
        let new_axes = process_axes(axes, self.layout.ndim())?;
        new_axes.iter().for_each(|&x| {
            res_shape = yield_one_before(&res_shape, x);
        });
        self.reshape(&res_shape)
    }

    pub fn permute(&self, axes: &[i64]) -> Result<Tensor, TensorError> {
        let permuted_layout = self.layout.permute(axes)?;
        Ok(Tensor {
            data: self.data.clone(),
            layout: permuted_layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
        })
    }

    pub fn permute_inv(&self, axes: &[i64]) -> Result<Tensor, TensorError> {
        let permuted_layout = self.layout.permute_inv(axes)?;
        Ok(Tensor {
            data: self.data.clone(),
            layout: permuted_layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
        })
    }

    pub fn expand(&self, shape: &[i64]) -> Result<Tensor, TensorError> {
        let res_shape = Shape::from(shape);
        let res_strides = self.layout.expand_strides(&res_shape)?;
        let res_layout = Layout::new(res_shape, res_strides);
        Ok(Tensor {
            data: self.data.clone(),
            layout: res_layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
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
        Ok(Tensor {
            data: ptr,
            parent: new_parent,
            mem_layout: self.mem_layout.clone(),
            layout: new_layout,
            backend: self.backend.clone(),
            dtype: self.dtype.clone(),
            device: self.device.clone(),
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
            (repeats.axes.len() - 1)
                .try_into()
                .expect("repeats.axes.len() - 1 is negative"),
            self.layout.ndim(),
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
        let val = if axes < 0 {
            (self.layout.ndim() as i64 + axes)
                .try_into()
                .expect("axis is still negative after adding ndim")
        } else {
            axes as usize
        };
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
        let axis1 = axis1
            .try_into()
            .expect("axis1 is still negative after adding ndim");
        let axis2 = axis2
            .try_into()
            .expect("axis2 is still negative after adding ndim");
        ShapeError::check_index_out_of_range(axis1, self.layout.ndim())?;
        ShapeError::check_index_out_of_range(axis2, self.layout.ndim())?;
        let mut new_shape = self.layout.shape().to_vec();
        let mut new_strides = self.layout.strides().to_vec();
        new_shape.swap(axis1, axis2);
        new_strides.swap(axis1, axis2);
        let layout = Layout::new(new_shape, new_strides);
        Ok(Tensor {
            data: self.data.clone(),
            layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
        })
    }

    pub fn flatten(&self, mut start_dim: i64, mut end_dim: i64) -> Result<Tensor, TensorError> {
        if start_dim < 0 {
            start_dim += self.layout.ndim() as i64;
        }
        if end_dim < 0 {
            end_dim += self.layout.ndim() as i64;
        }
        let start = start_dim
            .try_into()
            .expect("start_dim is still negative after adding ndim");
        let end = end_dim
            .try_into()
            .expect("end_dim is still negative after adding ndim");
        let shape = self.layout.shape();
        ShapeError::check_index_out_of_range(start, self.layout.ndim())?;
        ShapeError::check_index_out_of_range(end, self.layout.ndim())?;
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
        let broadcasted_layout = self.layout.broadcast_to(&res_shape)?;
        Ok(Tensor {
            data: self.data.clone(),
            layout: broadcasted_layout,
            dtype: self.dtype.clone(),
            device: self.device.clone(),
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
        })
    }

    #[track_caller]
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
            let layout = Layout::new(res_shape, res_strides);
            let new_parent = if self.parent.is_none() {
                Some(self.data.clone())
            } else {
                self.parent.clone()
            };
            Ok(Tensor {
                data: Pointer::new(res_ptr, len),
                layout,
                dtype: self.dtype.clone(),
                device: self.device.clone(),
                parent: new_parent,
                mem_layout: self.mem_layout.clone(),
                backend: self.backend.clone(),
            })
        }
        #[cfg(not(feature = "bound_check"))]
        {
            let layout = Layout::new(res_shape, res_strides);
            let new_parent = if self.parent.is_none() {
                Some(self.data.clone())
            } else {
                self.parent.clone()
            };
            Ok(Tensor {
                data: Pointer::new(res_ptr, 0),
                layout,
                dtype: self.dtype.clone(),
                device: self.device.clone(),
                parent: new_parent,
                mem_layout: self.mem_layout.clone(),
                backend: self.backend.clone(),
            })
        }
    }

    pub fn concat(
        tensors: Vec<&Self>,
        axis: i64,
        keepdims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let length = tensors.len();
        let dtype = tensors[0].dtype;
        let device = &tensors[0].device;

        let ndim = tensors[0].ndim();

        for i in tensors.iter() {
            if i.ndim() != ndim {
                return Err(ShapeError::ConcatDimMismatch {
                    expected: ndim,
                    actual: i.ndim(),
                    location: Location::caller(),
                }
                .into());
            }
        }
        let axis = if axis < 0 {
            (axis + ndim as i64).try_into().expect("axis out of range")
        } else {
            axis as usize
        };
        for i in tensors.iter() {
            if &i.device != device {
                panic!("concat tensors must be on the same device");
            }
            if i.dtype != dtype {
                panic!("concat tensors must have the same dtype");
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
        let new_tensor = Self::empty(&new_shape, dtype, device.clone())?;
        let mut begin = 0;
        let mut res_slices = Vec::with_capacity(length);
        for i in tensors.iter() {
            let mut selections = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); new_shape.len()];
            selections[axis] = (begin, begin + i.shape()[axis], 1);
            begin += i.shape()[axis];
            let res_tensor = new_tensor.slice(&selections)?;
            res_slices.push(res_tensor);
        }
        let tensors = tensors.iter().map(|x| (*x).clone()).collect::<Vec<Self>>();
        let num_threads = length.min(current_num_threads());
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
                for (mut res, inp) in res_tensors.into_iter().zip(inputs.into_iter()) {
                    res._copy_from(&inp, 1);
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
            new_tensor.reshape(&res_shape)
        } else {
            Ok(new_tensor)
        }
    }
}
