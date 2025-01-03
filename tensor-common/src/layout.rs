use std::panic::Location;

use crate::{
    axis::{process_axes, Axis},
    error::{base::TensorError, shape::ShapeError},
    shape::Shape,
    shape_utils::{is_reshape_possible, predict_broadcast_shape},
    strides::Strides,
    strides_utils::shape_to_strides,
};

/// `Layout` stores the `shape` and `strides` of a tensor
///
/// this struct is being used `internally` by the library
///
/// it is also widely being used to perform shape and strides related operations such as `reshape`, `permute`, `broadcast`, `reduce`, etc.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Layout {
    /// the shape of the tensor
    shape: Shape,
    /// the strides of the tensor
    strides: Strides,
}

impl Layout {
    /// create a new `Layout` instance
    ///
    /// # Arguments
    ///
    /// * `shape` - any type implemented `Into<Shape> for T` or `From<T> for Shape`
    /// * `strides` - any type implemented `Into<Strides> for T` or `From<T> for Strides`
    ///
    /// # Panics
    ///
    /// if the length of `shape` and `strides` is not equal
    pub fn new<A: Into<Shape>, B: Into<Strides>>(shape: A, strides: B) -> Self {
        let shape = shape.into();
        let strides = strides.into();
        assert_eq!(shape.len(), strides.len());
        Layout { shape, strides }
    }

    /// Returns the shape of the layout
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Set the shape of the layout
    pub fn set_shape(&mut self, shape: Shape) {
        self.shape = shape;
    }

    /// Returns the strides of the layout
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Set the strides of the layout
    pub fn set_strides(&mut self, strides: Strides) {
        self.strides = strides;
    }

    /// Returns the number of dimensions of the layout
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

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
    /// use tensor_common::layout::Layout;
    /// use tensor_common::shape::Shape;
    /// use tensor_common::strides::Strides;
    ///
    /// let shape = Shape::from(vec![2, 3, 4]);
    /// let strides = Strides::from(vec![12, 4, 1]);
    ///
    /// let layout = Layout::new(shape.clone(), strides.clone());
    ///
    /// let new_shape = Shape::from(vec![3, 2, 4]);
    /// let new_strides = layout.is_reshape_possible(&new_shape).unwrap();
    ///
    /// assert_eq!(new_strides, Strides::from(vec![8, 4, 1]));
    /// ```
    pub fn is_reshape_possible(&self, shape: &[i64]) -> Option<Strides> {
        if self.size() != shape.iter().product::<i64>() {
            return None;
        }
        is_reshape_possible(&self.shape, &self.strides, shape)
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
    /// * `Strides` - the new strides after expanding
    pub fn expand_strides(&self, expand_shape: &[i64]) -> Strides {
        let mut res_strides = vec![0; expand_shape.len()];
        expand_shape
            .iter()
            .enumerate()
            .rev()
            .zip(self.shape.iter().rev())
            .zip(self.strides.iter().rev())
            .for_each(|(((idx, new_dim), old_dim), old_stride)| {
                if new_dim != old_dim && old_dim == &1 {
                    res_strides[idx] = 0;
                } else {
                    res_strides[idx] = *old_stride;
                }
            });
        res_strides.into()
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
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn permute<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Layout, TensorError> {
        let axes = process_axes(axes, self.shape.len())?;
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
    pub fn permute_inv<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Layout, TensorError> {
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
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn inplace_reshape(&self, shape: &Shape) -> std::result::Result<Layout, TensorError> {
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
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn broadcast(&self, other: &Layout) -> std::result::Result<Layout, TensorError> {
        let shape = predict_broadcast_shape(&self.shape, &other.shape)?;
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
    pub fn reduce<A: Into<Axis>>(
        &self,
        axes: A,
        keep_dims: bool,
    ) -> std::result::Result<Layout, TensorError> {
        let axis = process_axes(axes, self.shape.len())?;
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
}

// Implementing the From trait for the `Layout` struct, when the user pass any of the following types, it will be converted to `Layout` automatically

impl From<Shape> for Layout {
    /// internally, it will call `shape_to_strides` to calculate the strides
    ///
    /// # See Also
    /// - [shape_to_strides](crate::strides_utils::shape_to_strides)
    fn from(shape: Shape) -> Self {
        let strides = shape_to_strides(&shape);
        Layout { shape, strides }
    }
}

impl From<&Shape> for Layout {
    /// internally, it will call `shape_to_strides` to calculate the strides
    ///
    /// # See Also
    /// - [shape_to_strides](crate::strides_utils::shape_to_strides)
    fn from(shape: &Shape) -> Self {
        let strides = shape_to_strides(shape);
        Layout {
            shape: shape.clone(),
            strides,
        }
    }
}

impl From<(Shape, Strides)> for Layout {
    fn from((shape, strides): (Shape, Strides)) -> Self {
        Layout { shape, strides }
    }
}

impl From<(Shape, Vec<i64>)> for Layout {
    fn from((shape, strides): (Shape, Vec<i64>)) -> Self {
        Layout {
            shape,
            strides: strides.into(),
        }
    }
}

impl From<(&Shape, Vec<i64>)> for Layout {
    fn from((shape, strides): (&Shape, Vec<i64>)) -> Self {
        Layout {
            shape: shape.into(),
            strides: strides.into(),
        }
    }
}

impl From<(&Shape, &[i64])> for Layout {
    fn from((shape, strides): (&Shape, &[i64])) -> Self {
        Layout {
            shape: shape.into(),
            strides: strides.into(),
        }
    }
}

impl From<&(Shape, Strides)> for Layout {
    fn from((shape, strides): &(Shape, Strides)) -> Self {
        Layout {
            shape: shape.clone(),
            strides: strides.clone(),
        }
    }
}

impl From<&Layout> for Layout {
    fn from(layout: &Layout) -> Self {
        Layout {
            shape: layout.shape.clone(),
            strides: layout.strides.clone(),
        }
    }
}

impl From<(&Shape, &Strides)> for Layout {
    fn from((shape, strides): (&Shape, &Strides)) -> Self {
        Layout {
            shape: shape.clone(),
            strides: strides.clone(),
        }
    }
}
