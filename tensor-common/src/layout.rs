use serde::Serialize;
use serde::ser::SerializeStruct;
use crate::{
    axis::{ process_axes, Axis },
    err_handler::ErrHandler,
    shape::Shape,
    shape_utils::{ is_reshape_possible, predict_broadcast_shape },
    strides::Strides,
    strides_utils::{ shape_to_strides, strides_is_contiguous },
};

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Shape,
    strides: Strides,
}

impl Serialize for Layout {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer {
        let mut state = serializer.serialize_struct("Layout", 2)?;
        state.serialize_field("shape", &self.shape.inner())?;
        state.serialize_field("strides", &self.strides.inner())?;
        state.end()
    }
}

impl Layout {
    pub fn new<A: Into<Shape>, B: Into<Strides>>(shape: A, strides: B) -> Self {
        let shape = shape.into();
        let strides = strides.into();
        assert_eq!(shape.len(), strides.len());
        Layout { shape, strides }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn set_shape(&mut self, shape: Shape) {
        self.shape = shape;
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    pub fn set_strides(&mut self, strides: Strides) {
        self.strides = strides;
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_reshape_possible(&self, shape: &Shape) -> Option<Strides> {
        is_reshape_possible(&self.shape, &self.strides, shape)
    }

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
        return (self.shape[max_idx] * max_stride) as usize;
    }

    pub fn permute<A: Into<Axis>>(&self, axes: A) -> anyhow::Result<Layout> {
        let axes = process_axes(axes, self.shape.len())?;
        ErrHandler::check_ndim_match(axes.len(), self.shape.len())?;
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

    pub fn inplace_reshape(&self, shape: &Shape) -> anyhow::Result<Layout> {
        if let Some(new_strides) = self.is_reshape_possible(shape) {
            Ok(Layout {
                shape: shape.clone(),
                strides: new_strides,
            })
        } else {
            return Err(
                ErrHandler::InplaceReshapeError(
                    format!("cannot perform inplace reshape {:?} to {:?}.", self.shape, shape)
                ).into()
            );
        }
    }

    pub fn swap_axis(&self, mut axis1: i64, mut axis2: i64) -> anyhow::Result<Layout> {
        ErrHandler::check_axis_same(axis1, axis2)?;
        if axis1 < 0 {
            while axis1 < 0 {
                axis1 += self.shape.len() as i64;
            }
        }
        if axis2 < 0 {
            while axis2 < 0 {
                axis2 += self.shape.len() as i64;
            }
        }
        ErrHandler::check_index_in_range(self.ndim(), axis1 as usize)?;
        ErrHandler::check_index_in_range(self.ndim(), axis2 as usize)?;
        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();
        new_shape.swap(axis1 as usize, axis2 as usize);
        new_strides.swap(axis1 as usize, axis2 as usize);
        Ok(Layout::new(new_shape, new_strides))
    }

    pub fn broadcast(&self, other: &Layout) -> anyhow::Result<Layout> {
        let shape = predict_broadcast_shape(&self.shape, &other.shape)?;
        let strides = shape_to_strides(&shape);
        Ok(Layout { shape, strides })
    }

    pub fn reduce<A: Into<Axis>>(&self, axes: A, keep_dims: bool) -> anyhow::Result<Layout> {
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
        let new_strides = shape_to_strides(&new_shape);
        Ok(Layout {
            shape: new_shape.into(),
            strides: new_strides,
        })
    }

    pub fn size(&self) -> i64 {
        self.shape.iter().product::<i64>()
    }

    pub fn is_contiguous(&self) -> bool {
        strides_is_contiguous(&self.strides)
    }

    pub fn continuous_strides(&self) -> Strides {
        shape_to_strides(&self.shape)
    }
}

impl From<Shape> for Layout {
    fn from(shape: Shape) -> Self {
        let strides = shape_to_strides(&shape);
        Layout { shape, strides }
    }
}

impl From<&Shape> for Layout {
    fn from(shape: &Shape) -> Self {
        let strides = shape_to_strides(&shape);
        Layout { shape: shape.clone(), strides }
    }
}
