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

impl<T> Scalar<T>
where
    T: TypeCommon,
{
    /// Create a new scalar
    pub fn new(val: String) -> Self {
        Self {
            val,
            _marker: PhantomData,
        }
    }
    /// Get the value of the scalar
    pub fn val(&self) -> &str {
        &self.val
    }
}

impl std::ops::Add for Scalar<bool> {
    type Output = Scalar<bool>;

    fn add(self, rhs: Self) -> Self::Output {
        Scalar::new(format!("({} || {})", self.val, rhs.val))
    }
}

impl Scalar<half::f16> {
    /// add two half::f16 scalars and return a f32 scalar
    pub fn add_ret_f32(self, rhs: Self) -> Scalar<f32> {
        Scalar::new(format!(
            "(__half2float({}) + __half2float({}))",
            self.val, rhs.val
        ))
    }
    /// subtract two half::f16 scalars and return a f32 scalar
    pub fn sub_ret_f32(self, rhs: Self) -> Scalar<f32> {
        Scalar::new(format!(
            "(__half2float({}) - __half2float({}))",
            self.val, rhs.val
        ))
    }
    /// multiply two half::f16 scalars and return a f32 scalar
    pub fn mul_ret_f32(self, rhs: Self) -> Scalar<f32> {
        Scalar::new(format!(
            "(__half2float({}) * __half2float({}))",
            self.val, rhs.val
        ))
    }
}
