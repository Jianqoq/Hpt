use std::{fmt::Display, marker::PhantomData};

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
    T: TypeCommon + Display,
{
    /// Create a new scalar
    pub fn new<U: ToString>(val: U) -> Self {
        Self {
            val: val.to_string(),
            _marker: PhantomData,
        }
    }

    /// Create a new scalar from a value
    pub fn new_from_val(val: T) -> Self {
        match T::STR {
            "bool" | "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64" | "isize"
            | "usize" => Self {
                val: format!("{}", val),
                _marker: PhantomData,
            },
            "bf16" => Self {
                val: format!("__nv_bfloat16((unsigned short){})", val),
                _marker: PhantomData,
            },
            "f16" => Self {
                val: format!("__half((unsigned short){})", val),
                _marker: PhantomData,
            },
            "f32" => Self {
                val: format!("(float){}", val),
                _marker: PhantomData,
            },
            "f64" => Self {
                val: format!("(double){}", val),
                _marker: PhantomData,
            },
            "c32" => Self {
                val: format!("make_cuFloatComplex((float){}, (float)0.0)", val),
                _marker: PhantomData,
            },
            "c64" => Self {
                val: format!("make_cuDoubleComplex((double){}, (double)0.0)", val),
                _marker: PhantomData,
            },
            _ => panic!("Unsupported type: {}", T::STR),
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
    /// equal
    pub(crate) fn __eq(self, rhs: Self) -> Scalar<bool> {
        Scalar::new(format!("({} == {})", self.val, rhs.val))
    }
    /// not equal
    pub(crate) fn __ne(self, rhs: Self) -> Scalar<bool> {
        Scalar::new(format!("({} != {})", self.val, rhs.val))
    }
    /// less than
    pub(crate) fn __lt(self, rhs: Self) -> Scalar<bool> {
        Scalar::new(format!("({} < {})", self.val, rhs.val))
    }
    /// less than or equal
    pub(crate) fn __le(self, rhs: Self) -> Scalar<bool> {
        Scalar::new(format!("({} <= {})", self.val, rhs.val))
    }
    /// greater than
    pub(crate) fn __gt(self, rhs: Self) -> Scalar<bool> {
        Scalar::new(format!("({} > {})", self.val, rhs.val))
    }
    /// greater than or equal
    pub(crate) fn __ge(self, rhs: Self) -> Scalar<bool> {
        Scalar::new(format!("({} >= {})", self.val, rhs.val))
    }
}
