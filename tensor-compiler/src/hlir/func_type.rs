use std::fmt::Display;

use tensor_types::dtype::Dtype;

use super::{expr::Expr, exprs::{ TensorType, Tuple }};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Type {
    Dtype(Dtype),
    Tensor(TensorType),
    Tuple(Tuple),
    Ptr(Dtype),
    Str,
    None,
}

impl Type {
    pub fn make_dtype(dtype: Dtype) -> Self {
        Type::Dtype(dtype)
    }
    pub fn make_tensor(tensor: TensorType) -> Self {
        Type::Tensor(tensor)
    }
    pub fn make_tuple(tuple: Tuple) -> Self {
        Type::Tuple(tuple)
    }
    pub fn make_str() -> Self {
        Type::Str
    }
    pub fn make_none() -> Self {
        Type::None
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Dtype(dtype) => write!(f, "{}", dtype),
            Type::Tensor(tensor) => write!(f, "{}", tensor),
            Type::Tuple(tuple) => write!(f, "{}", tuple),
            Type::Str => write!(f, "str"),
            Type::Ptr(dtype) => write!(f, "*{}", dtype),
            Type::None => write!(f, "()"),
        }
    }
}

impl Into<Expr> for Type {
    fn into(self) -> Expr {
        Expr::Type(self)
    }
}

impl Into<Expr> for &Type {
    fn into(self) -> Expr {
        Expr::Type(self.clone())
    }
}