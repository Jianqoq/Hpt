use crate::{
    shape::shape::Shape, strides::strides::Strides, strides::strides_utils::shape_to_strides,
};

/// `Layout` stores the `shape` and `strides` of a tensor
///
/// this struct is being used `internally` by the library
///
/// it is also widely being used to perform shape and strides related operations such as `reshape`, `permute`, `broadcast`, `reduce`, etc.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Layout {
    /// the shape of the tensor
    pub(crate) shape: Shape,
    /// the strides of the tensor
    pub(crate) strides: Strides,
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

    /// Returns the size of the outer loop
    pub fn outer_loop_size(&self) -> i64 {
        let inner_loop_size = *self.shape.last().unwrap_or(&1);
        self.size() / inner_loop_size
    }

    /// Returns the size of the inner loop
    pub fn inner_loop_size(&self) -> i64 {
        *self.shape.last().unwrap_or(&1)
    }

    /// Returns the most inner dimension's stride
    pub fn last_stride(&self) -> i64 {
        *self.strides.last().unwrap_or(&0)
    }
}

// Implementing the From trait for the `Layout` struct, when the user pass any of the following types, it will be converted to `Layout` automatically

impl From<Shape> for Layout {
    /// internally, it will call `shape_to_strides` to calculate the strides
    ///
    /// # See Also
    /// - [shape_to_strides](crate::strides::strides_utils::shape_to_strides)
    fn from(shape: Shape) -> Self {
        let strides = shape_to_strides(&shape);
        Layout { shape, strides }
    }
}

impl From<&Shape> for Layout {
    /// internally, it will call `shape_to_strides` to calculate the strides
    ///
    /// # See Also
    /// - [shape_to_strides](crate::strides::strides_utils::shape_to_strides)
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
