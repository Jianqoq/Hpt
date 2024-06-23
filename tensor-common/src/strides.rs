use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use serde::{Deserialize, Serialize};

use crate::strides_utils::strides_is_contiguous;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Strides {
    pub(crate) inner: Arc<Vec<i64>>,
}

impl Strides {
    pub fn inner(&self) -> &Vec<i64> {
        &self.inner
    }
}

#[derive(Serialize, Deserialize)]
pub struct StridesHelper {
    pub(crate) inner: Vec<i64>,
}

impl Serialize for Strides {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        StridesHelper {
            inner: self.inner.as_ref().clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Strides {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = StridesHelper::deserialize(deserializer)?;
        Ok(Strides {
            inner: helper.inner.into(),
        })
    }
}

impl std::fmt::Debug for Strides {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Strides {{ {:?} }}", self.inner))
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

impl Strides {
    pub fn new<S: IntoStrides>(v: S) -> Self {
        v.into_strides()
    }

    pub fn is_countiguous(&self) -> bool {
        strides_is_contiguous(&self.inner)
    }
}
