use std::{
    fmt::Display,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use crate::{strides::Strides, strides_utils::shape_to_strides};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};

/// Represents the shape of a multi-dimensional structure, such as a tensor or an array.
///
/// `Shape` is an enum currently consisting of a single variant, `Vec`, which encapsulates
/// a vector of `i64` values. These values typically represent the size of each dimension
/// in a multi-dimensional array or tensor. The use of `Arc<Vec<i64>>` allows for efficient
/// cloning and sharing of shape information without duplicating the underlying data.
///
/// # Note
/// User don't need to use it directly, the convertion happens right after the user passes the shape data to the functions.
///
/// # Variants
///
/// - `Vec(Arc<Vec<i64>>)` - Encapsulates a dynamically-sized list of dimensions. Each `i64`
///   value in the vector represents the size of a dimension in the multi-dimensional structure.
///
/// # Examples
///
/// Creating a `Shape` from a vector of i64:
/// ```
/// use tensor_common::Shape;
/// let shape_vec: Vec<i64> = vec![3, 4, 5];
/// let shape = Shape::from(shape_vec);
/// ```
///
/// Creating a `Shape` from a fixed-size array of i64:
/// ```
/// use tensor_common::Shape;
/// let shape_array = [3, 4, 5];
/// let shape = Shape::from(shape_array);
/// ```
///
/// Creating a `Shape` from a slice of i64:
/// ```
/// use tensor_common::Shape;
/// let shape_slice: &[i64] = &[3, 4, 5];
/// let shape = Shape::from(shape_slice);
/// ```
///
/// Creating a `Shape` from an `Arc<Vec<i64>>`:
/// ```
/// use std::sync::Arc;
/// use tensor_common::Shape;
/// let shape_arc_vec: Arc<Vec<i64>> = Arc::new(vec![3, 4, 5]);
/// let shape = Shape::from(shape_arc_vec);
/// ```
///
/// Creating a `Shape` from an `Arc<[i64]>`:
/// ```
/// use std::sync::Arc;
/// use tensor_common::Shape;
/// let shape_arc_array: Arc<[i64; 3]> = Arc::new([3, 4, 5]);
/// let shape = Shape::from(shape_arc_array);
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    inner: Arc<Vec<i64>>,
}

impl Serialize for Shape {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Shape", 1)?;
        state.serialize_field("inner", self.inner.as_ref())?;
        state.end()
    }
}

#[derive(Serialize, Deserialize)]
pub struct ShapeHelper {
    inner: Vec<i64>,
}

impl Shape {
    pub fn new<S: Into<Shape>>(shape: S) -> Self {
        shape.into()
    }

    pub fn size(&self) -> i64 {
        self.iter().product()
    }

    pub fn to_vec(&self) -> Vec<i64> {
        self.inner.as_ref().clone()
    }

    pub fn inner(&self) -> &Vec<i64> {
        &self.inner
    }

    /// directly calculate the strides based on the shape
    /// # Return
    /// - `Strides` - contiguous strides of the shape
    pub fn to_strides(&self) -> Strides {
        shape_to_strides(self)
    }

    pub fn sub_one(&self) -> Shape {
        self.iter().map(|x| *x - 1).collect::<Vec<i64>>().into()
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("shape({:?})", self.inner))
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("shape({:?})", self.inner))
    }
}

impl Default for Shape {
    fn default() -> Self {
        Shape {
            inner: Arc::new(Vec::new()),
        }
    }
}

impl Deref for Shape {
    type Target = Vec<i64>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Shape {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
}

// Implementing the From trait for the `Shape` struct, when the user pass any of the following types, it will be converted to `Shape` automatically

impl From<&Shape> for Shape {
    fn from(v: &Shape) -> Self {
        Shape {
            inner: Arc::clone(&v.inner),
        }
    }
}

impl From<Arc<Vec<i64>>> for Shape {
    fn from(v: Arc<Vec<i64>>) -> Self {
        Shape { inner: v }
    }
}

impl From<Vec<i64>> for Shape {
    fn from(v: Vec<i64>) -> Self {
        Shape { inner: Arc::new(v) }
    }
}

impl<const N: usize> From<[i64; N]> for Shape {
    fn from(v: [i64; N]) -> Self {
        Shape {
            inner: Arc::new(v.to_vec()),
        }
    }
}

impl<'a, const N: usize> From<&'a [i64; N]> for Shape {
    fn from(v: &'a [i64; N]) -> Self {
        Shape {
            inner: Arc::new(v.to_vec()),
        }
    }
}

impl<'a> From<&'a Vec<i64>> for Shape {
    fn from(v: &'a Vec<i64>) -> Self {
        Shape {
            inner: Arc::new(v.clone()),
        }
    }
}

impl<'a> From<&'a Arc<Vec<i64>>> for Shape {
    fn from(v: &'a Arc<Vec<i64>>) -> Self {
        Shape {
            inner: Arc::clone(v),
        }
    }
}

impl<const N: usize> From<Arc<[i64; N]>> for Shape {
    fn from(v: Arc<[i64; N]>) -> Self {
        Shape {
            inner: Arc::new(v.to_vec()),
        }
    }
}

impl From<&[i64]> for Shape {
    fn from(v: &[i64]) -> Self {
        Shape {
            inner: Arc::new(v.to_vec()),
        }
    }
}
