use std::sync::Arc;

use crate::float::Float;

#[derive(Clone, PartialEq, PartialOrd, Hash, Eq, Default)]
pub enum Op {
    Add,
    Sum {
        axes: Arc<Vec<usize>>,
    },
    Mean {
        axes: Arc<Vec<usize>>,
    },
    Var {
        axes: Arc<Vec<usize>>,
    },
    Max {
        axes: Arc<Vec<usize>>,
    },
    Min {
        axes: Arc<Vec<usize>>,
    },
    Sub,
    Mul,
    Div,
    Square,
    Sqrt,
    Matmul,
    Arange {
        start: Arc<Float>,
        end: Arc<Float>,
        step: Arc<Float>,
    },
    Ones,
    Zeros,
    Randn {
        mean: Arc<Float>,
        std: Arc<Float>,
    },
    Power,
    Reshape,
    Transpose {
        axes: Arc<Vec<usize>>,
    },
    Slice {
        offset: isize,
    },
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,
    Abs,
    Ceil,
    Exp,
    Floor,
    AsType,
    Expand,
    Contiguous,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    #[default]
    Null,
}