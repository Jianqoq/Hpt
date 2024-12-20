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
        match T::ID {
            crate::dtype::Dtype::Bool
            | crate::dtype::Dtype::I8
            | crate::dtype::Dtype::U8
            | crate::dtype::Dtype::I16
            | crate::dtype::Dtype::U16
            | crate::dtype::Dtype::I32
            | crate::dtype::Dtype::U32
            | crate::dtype::Dtype::I64
            | crate::dtype::Dtype::U64
            | crate::dtype::Dtype::Isize
            | crate::dtype::Dtype::Usize => Self {
                val: format!("{}", val),
                _marker: PhantomData,
            },
            crate::dtype::Dtype::BF16 => Self {
                val: format!("__nv_bfloat16((unsigned short){})", val),
                _marker: PhantomData,
            },
            crate::dtype::Dtype::F16 => Self {
                val: format!("__half((unsigned short){})", val),
                _marker: PhantomData,
            },
            crate::dtype::Dtype::F32 => Self {
                val: format!("(float){}", val),
                _marker: PhantomData,
            },
            crate::dtype::Dtype::F64 => Self {
                val: format!("(double){}", val),
                _marker: PhantomData,
            },
            crate::dtype::Dtype::C32 => Self {
                val: format!("make_cuFloatComplex((float){}, (float)0.0)", val),
                _marker: PhantomData,
            },
            crate::dtype::Dtype::C64 => Self {
                val: format!("make_cuDoubleComplex((double){}, (double)0.0)", val),
                _marker: PhantomData,
            },
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
