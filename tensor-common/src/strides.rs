use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

/// Represents the strides of a multi-dimensional structure, such as a tensor or an array.
///
/// Strides are used to calculate the memory offset of an element in a multi-dimensional structure.
///
/// # Example
/// for a strides of `[1, 2, 3]`, we can access the 1st dimension element by adding 1 memory offset, 
/// the 2nd dimension element by adding 2 memory offset, and the 3rd dimension element by adding 3 memory offset.
///
/// # Note
/// User don't need to use it directly, the convertion happens right after the user passes the strides data to the functions.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Strides {
    pub(crate) inner: Arc<Vec<i64>>,
}

impl Strides {
    pub fn inner(&self) -> &Vec<i64> {
        &self.inner
    }
    pub fn inner_(&self) -> &Arc<Vec<i64>> {
        &self.inner
    }
}

impl std::fmt::Debug for Strides {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("strides({:?})", self.inner))
    }
}

impl std::fmt::Display for Strides {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("strides({:?})", self.inner))
    }
}

pub trait IntoStrides {
    fn into_strides(self) -> Strides;
}

impl From<Arc<Vec<i64>>> for Strides {
    fn from(v: Arc<Vec<i64>>) -> Self {
        Strides { inner: v }
    }
}

impl From<Vec<i64>> for Strides {
    fn from(v: Vec<i64>) -> Self {
        Strides { inner: Arc::new(v) }
    }
}

impl From<&[i64]> for Strides {
    fn from(v: &[i64]) -> Self {
        Strides {
            inner: Arc::new(v.to_vec()),
        }
    }
}

impl<const N: usize> From<[i64; N]> for Strides {
    fn from(v: [i64; N]) -> Self {
        Strides {
            inner: Arc::new(v.to_vec()),
        }
    }
}

impl<'a, const N: usize> From<&'a [i64; N]> for Strides {
    fn from(v: &'a [i64; N]) -> Self {
        Strides {
            inner: Arc::new(v.to_vec()),
        }
    }
}

impl<'a> From<&'a Vec<i64>> for Strides {
    fn from(v: &'a Vec<i64>) -> Self {
        Strides {
            inner: Arc::new(v.clone()),
        }
    }
}

impl<'a> From<&'a Arc<Vec<i64>>> for Strides {
    fn from(v: &'a Arc<Vec<i64>>) -> Self {
        Strides {
            inner: Arc::clone(v),
        }
    }
}

impl<const N: usize> From<Arc<[i64; N]>> for Strides {
    fn from(v: Arc<[i64; N]>) -> Self {
        Strides {
            inner: Arc::new(v.to_vec()),
        }
    }
}

impl Default for Strides {
    fn default() -> Self {
        Strides {
            inner: Arc::new(Vec::new()),
        }
    }
}

impl Deref for Strides {
    type Target = Vec<i64>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Strides {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
}