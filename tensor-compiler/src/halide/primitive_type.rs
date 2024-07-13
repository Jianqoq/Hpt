use std::{ fmt::Display, sync::Arc };

use tensor_types::{ dtype::Dtype, type_promote::{ FloatOut, NormalOut } };

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum PrimitiveType {
    Dtype(Dtype),
    Tuple(Tuple),
    Array(Array),
    Ptr(Ptr),
    Str,
    Void,
}
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Tuple {
    pub inner: Arc<Vec<PrimitiveType>>,
}
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Array {
    pub inner: Arc<PrimitiveType>,
    pub size: i64,
}
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Ptr {
    pub inner: Arc<PrimitiveType>,
}

impl PrimitiveType {
    pub fn dtype(&self) -> Dtype {
        match self {
            PrimitiveType::Dtype(dtype) => *dtype,
            _ => unimplemented!(),
        }
    }
    pub fn _add(&self, other: &Self) -> Self {
        match (self, other) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._add(*rhs))
            }
            _ => unimplemented!(),
        }
    }
    pub fn _mul(&self, other: &Self) -> Self {
        match (self, other) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._mul(*rhs))
            }
            _ => unimplemented!(),
        }
    }
    pub fn _sub(&self, other: &Self) -> Self {
        match (self, other) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._sub(*rhs))
            }
            _ => unimplemented!(),
        }
    }
    pub fn _div(&self, other: &Self) -> Self {
        match (self, other) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._div(*rhs))
            }
            _ => unimplemented!(),
        }
    }
    pub fn _sin(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._sin()),
            _ => unimplemented!(),
        }
    }
    pub fn _cos(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._cos()),
            _ => unimplemented!(),
        }
    }
    pub fn _tan(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._tan()),
            _ => unimplemented!(),
        }
    }
    pub fn _asin(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._asin()),
            _ => unimplemented!(),
        }
    }
    pub fn _acos(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._acos()),
            _ => unimplemented!(),
        }
    }
    pub fn _atan(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._atan()),
            _ => unimplemented!(),
        }
    }
    pub fn _exp(&self) -> Self {
        match self {
            PrimitiveType::Dtype(dtype) => PrimitiveType::Dtype(dtype._exp()),
            _ => unimplemented!(),
        }
    }
    pub fn _log(&self, base: &Self) -> Self {
        match (self, base) {
            (PrimitiveType::Dtype(lhs), PrimitiveType::Dtype(rhs)) => {
                PrimitiveType::Dtype(lhs._log(*rhs))
            }
            _ => unimplemented!(),
        }
    }
}

impl Display for PrimitiveType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrimitiveType::Dtype(dtype) => write!(f, "{}", dtype),
            PrimitiveType::Tuple(tuple) => {
                if tuple.inner.is_empty() {
                    return write!(f, "()");
                }
                if tuple.inner.len() == 1 {
                    return write!(f, "({}, )", tuple.inner[0]);
                }
                write!(f, "({}", tuple.inner[0])?;
                for i in 1..tuple.inner.len() {
                    write!(f, ", {}", tuple.inner[i])?;
                }
                write!(f, ")")
            }
            PrimitiveType::Array(arr) => write!(f, "[{}; {}]", arr.inner, arr.size),
            PrimitiveType::Ptr(ptr) => write!(f, "*{}", ptr.inner),
            PrimitiveType::Str => write!(f, "str"),
            PrimitiveType::Void => write!(f, "void"),
        }
    }
}
