use std::panic::Location;

use crate::{
    axis::axis::{process_axes, Axis},
    error::{base::TensorError, shape::ShapeError},
    shape::{
        shape::Shape,
        shape_utils::{
            get_broadcast_axes_from, is_reshape_possible, predict_broadcast_shape, try_pad_shape,
        },
    },
    strides::{strides::Strides, strides_utils::shape_to_strides},
};

use super::layout::Layout;

impl Layout {
    /// # Internal Function
    ///
    /// a function mainly use for checking if the reshape is possible
    ///
    /// most of the case, a tensor can be reshaped if the dimension going to be reshaped is `contiguous`
    ///
    /// # Case
    ///
    /// if the shape is `[2, 3, 4, 5]` and the strides is `[60, 20, 5, 1]`, they are `contiguous`
    ///
    /// if we permute the shape to `[2, 3, 5, 4]` and the strides to `[60, 20, 1, 5]`, they are not `contiguous` generally, but the first two dimensions are `contiguous`
    ///
    /// if reshape is on the first two dimensions, like `[2, 3, 5, 4]` to `[6, 4, 5]`, or `[2, 3, 5, 4]` to `[3, 2, 5, 4]`, they are possible
    ///
    /// if reshape is on the last two dimensions, like `[2, 3, 5, 4]` to `[2, 3, 20]`, it is not possible
    ///
    /// # Arguments
    ///
    /// * `shape` - the new shape to be reshaped
    ///
    /// # Returns
    ///
    /// * `Option<Strides>` - if the reshape is possible, return the new strides, otherwise return `None`
    ///
    /// # Examples
    ///
    /// ```
    /// use hpt_common::layout::Layout;
    /// use hpt_common::shape::Shape;
    /// use hpt_common::strides::strides::Strides;
    ///
    /// let shape = Shape::from(vec![2, 3, 4]);
    /// let strides = strides::strides::from(vec![12, 4, 1]);
    ///
    /// let layout = Layout::new(shape.clone(), strides.clone());
    ///
    /// let new_shape = Shape::from(vec![3, 2, 4]);
    /// let new_strides = layout.is_reshape_possible(&new_shape).unwrap();
    ///
    /// assert_eq!(new_strides, strides::strides::from(vec![8, 4, 1]));
    /// ```
    pub fn is_reshape_possible(&self, shape: &[i64]) -> Option<Strides> {
        if self.size() != shape.iter().product::<i64>() {
            return None;
        }
        is_reshape_possible(&self.shape, &self.strides, shape)
    }

    /// # Internal Function
    ///
    /// a function use to calculate the broadcast layout based on the target shape
    ///
    /// # Arguments
    ///
    /// * `shape` - the target shape
    pub fn to_broadcast_layout(&self, target_shape: &[i64]) -> Result<Layout, TensorError> {
        let padded_shape = try_pad_shape(&self.shape, target_shape.len());

        // try_pad_shape can also used on strides
        let padded_strides = try_pad_shape(&self.strides, target_shape.len());
        let mut new_strides = vec![0; target_shape.len()];

        for i in 0..target_shape.len() {
            if padded_shape[i] == target_shape[i] {
                new_strides[i] = padded_strides[i];
            } else {
                new_strides[i] = 0;
            }
        }

        Ok(Layout {
            shape: target_shape.into(),
            strides: new_strides.into(),
        })
    }

    /// # Internal Function
    ///
    /// a function mainly use for expanding the strides of a tensor
    ///
    /// this function is simply convert the stride of the dimension going to be expanded to `0`
    ///
    /// # Arguments
    ///
    /// * `expand_shape` - the new shape to be expanded
    ///
    /// # Returns
    ///
    /// * `Result<Strides>` - the new strides after expanding
    pub fn expand_strides(&self, expand_shape: &[i64]) -> Result<Strides, TensorError> {
        let mut res_strides = vec![0; expand_shape.len()];
        for (((idx, new_dim), old_dim), old_stride) in expand_shape
            .iter()
            .enumerate()
            .rev()
            .zip(self.shape.iter().rev())
            .zip(self.strides.iter().rev())
        {
            if new_dim != old_dim && old_dim == &1 {
                res_strides[idx] = 0;
            } else if new_dim != old_dim && old_dim != &1 {
                return Err(ShapeError::ExpandError {
                    old_dim: *old_dim,
                    location: Location::caller(),
                }
                .into());
            } else {
                res_strides[idx] = *old_stride;
            }
        }
        Ok(res_strides.into())
    }

    /// # Internal Function
    /// a function mainly use for calculating the real size of a tensor
    /// pretty useful when the tensor is a view of another tensor
    pub fn real_size(&self) -> usize {
        assert_eq!(self.shape.len(), self.strides.len());
        let mut max_stride = 0;
        let mut max_idx = 0;
        for (idx, stride) in self.strides.iter().enumerate() {
            if *stride > max_stride {
                max_stride = *stride;
                max_idx = idx;
            }
        }
        (self.shape[max_idx] * max_stride) as usize
    }

    /// # Internal Function
    ///
    /// a function use to calculate the permuted layout
    ///
    /// # Arguments
    ///
    /// * `axes` - the new order of the dimensions
    ///
    /// # Returns
    ///
    /// * `Result<Layout>` - the new layout after permutation
    ///
    /// # Panics
    ///
    /// if the length of `axes` is not equal to the layout's ndim
    #[track_caller]
    pub fn permute<A: Into<Axis>>(&self, axes: A) -> Result<Layout, TensorError> {
        let axes = process_axes(axes, self.shape.len())?;
        let axes = if axes.len() == 0 {
            (0..self.shape.len()).rev().collect::<Vec<_>>()
        } else {
            axes
        };
        ShapeError::check_dim(axes.len(), self.shape.len())?;
        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();
        for i in axes.iter() {
            new_shape[*i] = self.shape()[axes[*i]];
            new_strides[*i] = self.strides()[axes[*i]];
        }
        Ok(Layout {
            shape: new_shape.into(),
            strides: new_strides.into(),
        })
    }

    /// # Internal Function
    ///
    /// a function use to calculate the inverse permuted layout
    ///
    /// # Arguments
    ///
    /// * `axes` - the new order of the dimensions
    ///
    /// # Returns
    ///
    /// * `Result<Layout>` - the new layout after inverse permutation
    pub fn permute_inv<A: Into<Axis>>(&self, axes: A) -> Result<Layout, TensorError> {
        let axes = process_axes(axes, self.shape.len())?;
        ShapeError::check_dim(axes.len(), self.shape.len())?;
        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();
        for i in axes.iter() {
            new_shape[axes[*i]] = self.shape()[*i];
            new_strides[axes[*i]] = self.strides()[*i];
        }
        Ok(Layout {
            shape: new_shape.into(),
            strides: new_strides.into(),
        })
    }

    /// # Internal Function
    ///
    /// perform an inplace reshape on the layout
    ///
    /// # Arguments
    ///
    /// * `shape` - the new shape to be reshaped
    ///
    /// # Returns
    ///
    /// * `Result<Layout>` - the new layout after reshape
    ///
    /// # Panics
    ///
    /// if the reshape is not possible
    #[track_caller]
    pub fn inplace_reshape(&self, shape: &Shape) -> Result<Layout, TensorError> {
        if let Some(new_strides) = self.is_reshape_possible(shape) {
            Ok(Layout {
                shape: shape.clone(),
                strides: new_strides,
            })
        } else {
            Err(ShapeError::InplaceReshapeError {
                message: "Inplace reshape is not possible".to_string(),
                location: Location::caller(),
            }
            .into())
        }
    }

    ///
    #[track_caller]
    pub fn broadcast_to(&self, other: &Shape) -> Result<Layout, TensorError> {
        let self_shape = try_pad_shape(self.shape(), other.len());
        let axes_to_broadcast =
            get_broadcast_axes_from(&self_shape, &other).expect("Cannot broadcast shapes");

        let mut new_strides = vec![0; other.len()];
        new_strides
            .iter_mut()
            .rev()
            .zip(self.strides().iter().rev())
            .for_each(|(a, b)| {
                *a = *b;
            });
        for &axis in axes_to_broadcast.iter() {
            assert_eq!(self_shape[axis], 1);
            new_strides[axis] = 0;
        }
        Ok(Layout {
            shape: other.clone(),
            strides: new_strides.into(),
        })
    }

    /// # Internal Function
    ///
    /// broadcast the layout to another layout
    ///
    /// # Arguments
    ///
    /// * `other` - the other layout to be broadcasted
    ///
    /// # Returns
    ///
    /// * `Result<Layout>` - the new layout after broadcast
    ///
    /// # Panics
    ///
    /// if the broadcast is not possible
    #[track_caller]
    pub fn broadcast(&self, res_shape: &Shape) -> Result<Layout, TensorError> {
        let shape = predict_broadcast_shape(&self.shape, res_shape)?;
        let strides = shape_to_strides(&shape);
        Ok(Layout { shape, strides })
    }

    /// # Internal Function
    ///
    /// reduce the layout to another layout
    ///
    /// this is mainly used for reducing the dimension of a tensor
    ///
    /// # Arguments
    ///
    /// * `axes` - the axes to be reduced
    ///
    /// * `keep_dims` - whether to keep the reduced dimensions
    ///
    /// # Returns
    ///
    /// * `Result<Layout>` - the new layout after reduction
    ///
    /// # Panics
    ///
    /// if the `axes` contains the same axis
    ///
    /// if the `axes` contains the same axis as the layout's ndim
    ///
    /// if the `axes` contains the axis out of range
    pub fn reduce<A: Into<Axis>>(&self, axes: A, keep_dims: bool) -> Result<Layout, TensorError> {
        let a: Axis = axes.into();
        let axis = process_axes(a, self.shape.len())?;
        let new_shape = if keep_dims {
            let mut vec = Vec::with_capacity(self.shape.len());
            for i in 0..self.shape.len() {
                if axis.contains(&i) {
                    vec.push(1);
                } else {
                    vec.push(self.shape[i]);
                }
            }
            vec
        } else {
            let mut vec = Vec::with_capacity(self.shape.len() - axis.len());
            for i in 0..self.shape.len() {
                if !axis.contains(&i) {
                    vec.push(self.shape[i]);
                }
            }
            vec
        };
        if new_shape.len() > 0 {
            let new_strides = shape_to_strides(&new_shape);
            Ok(Layout {
                shape: new_shape.into(),
                strides: new_strides,
            })
        } else {
            Ok(Layout {
                shape: vec![1].into(),
                strides: vec![1].into(),
            })
        }
    }

    /// simply return the product of the shape
    ///
    /// # Safety
    ///
    /// when the layout is a view of another layout, the size will be different, this method won't work
    #[inline(always)]
    pub fn size(&self) -> i64 {
        self.shape.iter().product::<i64>()
    }

    /// # Internal Function
    ///
    /// check if the layout is contiguous
    ///
    /// # Returns
    ///
    /// * `bool` - whether the layout is contiguous
    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for (&dim_size, &stride) in self.shape.iter().rev().zip(self.strides.iter().rev()) {
            if dim_size == 0 {
                continue;
            }
            if stride != expected_stride {
                return false;
            }
            expected_stride *= dim_size;
        }
        true
    }

    /// # Internal Function
    ///
    /// coalesce the dimensions of a layout
    ///
    /// # Returns
    ///
    /// * `Vec<Vec<usize>>` - the coalesced dimensions
    pub fn coalesce_dims(&self) -> Vec<Vec<usize>> {
        let shape = &self.shape;
        let strides = &self.strides;
        let mut groups = vec![vec![shape.len() - 1]];
        let mut current_stride = strides[shape.len() - 1];
        let mut current_size = shape[shape.len() - 1];

        for i in (0..shape.len() - 1).rev() {
            let expected_stride = current_stride * current_size;

            if strides[i] == expected_stride {
                groups.last_mut().unwrap().push(i);
            } else {
                groups.push(vec![i]);
            }

            current_stride = strides[i];
            current_size = shape[i];
        }

        for group in groups.iter_mut() {
            group.reverse();
        }
        groups.reverse();

        groups
    }
}
