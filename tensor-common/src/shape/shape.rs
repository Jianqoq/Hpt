use std::{ fmt::Display, ops::{ Deref, DerefMut }, sync::Arc };

use serde::{Deserialize, Serialize};

use crate::{ strides::strides::Strides, strides::strides_utils::shape_to_strides };

/// Represents the shape of a multi-dimensional structure, such as a tensor or an array.
///
/// # Note
/// User don't need to use it directly, the convertion happens right after the user passes the shape data to the functions.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    /// inner data of the shape
    inner: Arc<Vec<i64>>,
}

impl Shape {
    /// Create a new shape
    pub fn new<S: Into<Shape>>(shape: S) -> Self {
        shape.into()
    }

    /// Returns the size of the shape (product of all elements)
    pub fn size(&self) -> i64 {
        self.iter().product()
    }

    /// convert Shape to Vec<i64>
    pub fn to_vec(&self) -> Vec<i64> {
        self.inner.as_ref().clone()
    }

    /// Returns inner &Vec<i64>
    pub fn inner(&self) -> &Vec<i64> {
        &self.inner
    }

    /// directly calculate the strides based on the shape
    /// # Return
    /// - `Strides` - contiguous strides of the shape
    pub fn to_strides(&self) -> Strides {
        shape_to_strides(self)
    }

    /// Returns a new Shape with all elements decreased by 1
    pub fn sub_one(&self) -> Shape {
        self.iter()
            .map(|x| *x - 1)
            .collect::<Vec<i64>>()
            .into()
    }

    /// Returns a pointer to the inner data
    pub fn as_ptr(&self) -> *const i64 {
        self.inner.as_ptr()
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

impl<'a, const N: usize> From<&'a [usize; N]> for Shape {
    fn from(v: &'a [usize; N]) -> Self {
        Shape {
            inner: Arc::new(
                v
                    .into_iter()
                    .map(|x| *x as i64)
                    .collect()
            ),
        }
    }
}

impl<'a, const N: usize> From<&'a [i32; N]> for Shape {
    fn from(v: &'a [i32; N]) -> Self {
        Shape {
            inner: Arc::new(
                v
                    .into_iter()
                    .map(|x| *x as i64)
                    .collect()
            ),
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
