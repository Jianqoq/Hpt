use std::marker::PhantomData;

use crate::dtype::TypeCommon;

/// A scalar type for cuda
#[derive(Clone)]
pub struct Scalar<T>
where
    T: TypeCommon,
{
    pub(crate) val: String,
    _marker: PhantomData<T>,
}

impl<T> std::fmt::Display for Scalar<T>
where
    T: TypeCommon,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.val)
    }
}

impl<T> Scalar<T>
where
    T: TypeCommon,
{
    /// Create a new scalar
    pub fn new<U: ToString>(val: U) -> Self {
        Self {
            val: val.to_string(),
            _marker: PhantomData,
        }
    }
    /// Get the value of the scalar
    pub fn val(&self) -> &str {
        &self.val
    }

    /// assign a value to the scalar
    pub fn assign<U: TypeCommon>(mut self, val: Scalar<U>) -> Self {
        let val = format!("{} = {}", self.val, val.val);
        self.val = val;
        self
    }
}
