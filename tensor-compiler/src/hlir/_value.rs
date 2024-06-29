use std::fmt::Display;

use half::{ bf16, f16 };

#[derive(Clone, PartialEq, Debug)]
pub enum _Value {
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl From<u64> for _Value {
    fn from(v: u64) -> Self {
        _Value::Uint(v)
    }
}

impl From<&u64> for _Value {
    fn from(v: &u64) -> Self {
        _Value::Uint(*v)
    }
}

impl From<i64> for _Value {
    fn from(v: i64) -> Self {
        _Value::Int(v)
    }
}

impl From<&i64> for _Value {
    fn from(v: &i64) -> Self {
        _Value::Int(*v)
    }
}

impl From<f64> for _Value {
    fn from(v: f64) -> Self {
        _Value::Float(v)
    }
}

impl From<&f64> for _Value {
    fn from(v: &f64) -> Self {
        _Value::Float(*v)
    }
}

impl From<bool> for _Value {
    fn from(v: bool) -> Self {
        _Value::Bool(v)
    }
}

impl From<&bool> for _Value {
    fn from(v: &bool) -> Self {
        _Value::Bool(*v)
    }
}

impl From<u8> for _Value {
    fn from(v: u8) -> Self {
        _Value::Uint(v as u64)
    }
}

impl From<&u8> for _Value {
    fn from(v: &u8) -> Self {
        _Value::Uint(*v as u64)
    }
}

impl From<i8> for _Value {
    fn from(v: i8) -> Self {
        _Value::Int(v as i64)
    }
}

impl From<&i8> for _Value {
    fn from(v: &i8) -> Self {
        _Value::Int(*v as i64)
    }
}

impl From<f32> for _Value {
    fn from(v: f32) -> Self {
        _Value::Float(v as f64)
    }
}

impl From<&f32> for _Value {
    fn from(v: &f32) -> Self {
        _Value::Float(*v as f64)
    }
}

impl From<u16> for _Value {
    fn from(v: u16) -> Self {
        _Value::Uint(v as u64)
    }
}

impl From<&u16> for _Value {
    fn from(v: &u16) -> Self {
        _Value::Uint(*v as u64)
    }
}

impl From<i16> for _Value {
    fn from(v: i16) -> Self {
        _Value::Int(v as i64)
    }
}

impl From<&i16> for _Value {
    fn from(v: &i16) -> Self {
        _Value::Int(*v as i64)
    }
}

impl From<u32> for _Value {
    fn from(v: u32) -> Self {
        _Value::Uint(v as u64)
    }
}

impl From<&u32> for _Value {
    fn from(v: &u32) -> Self {
        _Value::Uint(*v as u64)
    }
}

impl From<i32> for _Value {
    fn from(v: i32) -> Self {
        _Value::Int(v as i64)
    }
}

impl From<&i32> for _Value {
    fn from(v: &i32) -> Self {
        _Value::Int(*v as i64)
    }
}

impl From<isize> for _Value {
    fn from(v: isize) -> Self {
        _Value::Int(v as i64)
    }
}

impl From<&isize> for _Value {
    fn from(v: &isize) -> Self {
        _Value::Int(*v as i64)
    }
}

impl From<usize> for _Value {
    fn from(v: usize) -> Self {
        _Value::Uint(v as u64)
    }
}

impl From<&usize> for _Value {
    fn from(v: &usize) -> Self {
        _Value::Uint(*v as u64)
    }
}

impl From<f16> for _Value {
    fn from(v: f16) -> Self {
        _Value::Float(f64::from(v))
    }
}

impl From<&f16> for _Value {
    fn from(v: &f16) -> Self {
        _Value::Float(f64::from(*v))
    }
}

impl From<bf16> for _Value {
    fn from(v: bf16) -> Self {
        _Value::Float(f64::from(v))
    }
}

impl From<&bf16> for _Value {
    fn from(v: &bf16) -> Self {
        _Value::Float(f64::from(*v))
    }
}

impl std::cmp::Eq for _Value {}

impl std::hash::Hash for _Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            _Value::Uint(v) => v.hash(state),
            _Value::Int(v) => v.hash(state),
            _Value::Float(v) => v.to_bits().hash(state),
            _Value::Bool(v) => v.hash(state),
        }
    }
}

impl Display for _Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            _Value::Uint(v) => write!(f, "{}", v),
            _Value::Int(v) => write!(f, "{}", v),
            _Value::Float(v) => write!(f, "{}", v),
            _Value::Bool(v) => write!(f, "{}", v),
        }
    }
}
