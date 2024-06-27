use std::sync::Arc;
use serde::ser::SerializeStruct;
use serde::Serialize;

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
    // Vec<Vec<usize>> means for each branch, each branch might return multiple nodes
    // this op is used to merge the results of multiple branches
    // only one branch contains the real data, other branches must return dead nodes,
    Merge,
    #[default]
    Null,
}

impl Serialize for Op {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer {
        match self {
            Op::Add => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Add")?;
                state.end()
            }
            Op::Sum { axes } => {
                let mut state = serializer.serialize_struct("Operation", 2)?;
                state.serialize_field("op", "Sum")?;
                state.serialize_field("axes", axes.as_ref())?;
                state.end()
            }
            Op::Mean { axes } => {
                let mut state = serializer.serialize_struct("Operation", 2)?;
                state.serialize_field("op", "Mean")?;
                state.serialize_field("axes", axes.as_ref())?;
                state.end()
            }
            Op::Var { axes } => {
                let mut state = serializer.serialize_struct("Operation", 2)?;
                state.serialize_field("op", "Var")?;
                state.serialize_field("axes", axes.as_ref())?;
                state.end()
            }
            Op::Max { axes } => {
                let mut state = serializer.serialize_struct("Operation", 2)?;
                state.serialize_field("op", "Max")?;
                state.serialize_field("axes", axes.as_ref())?;
                state.end()
            }
            Op::Min { axes } => {
                let mut state = serializer.serialize_struct("Operation", 2)?;
                state.serialize_field("op", "Min")?;
                state.serialize_field("axes", axes.as_ref())?;
                state.end()
            }
            Op::Sub => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Sub")?;
                state.end()
            }
            Op::Mul => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Mul")?;
                state.end()
            }
            Op::Div => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Div")?;
                state.end()
            }
            Op::Square => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Square")?;
                state.end()
            }
            Op::Sqrt => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Sqrt")?;
                state.end()
            }
            Op::Matmul => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Matmul")?;
                state.end()
            }
            Op::Arange { start, end, step } => {
                let mut state = serializer.serialize_struct("Operation", 4)?;
                state.serialize_field("op", "Arange")?;
                state.serialize_field("start", start.as_ref())?;
                state.serialize_field("end", end.as_ref())?;
                state.serialize_field("step", step.as_ref())?;
                state.end()
            }
            Op::Ones => todo!(),
            Op::Zeros => todo!(),
            Op::Randn { mean, std } => {
                let mut state = serializer.serialize_struct("Operation", 3)?;
                state.serialize_field("op", "Randn")?;
                state.serialize_field("mean", mean.as_ref())?;
                state.serialize_field("std", std.as_ref())?;
                state.end()
            }
            Op::Power => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Power")?;
                state.end()
            }
            Op::Reshape => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Reshape")?;
                state.end()
            }
            Op::Transpose { axes } => {
                let mut state = serializer.serialize_struct("Operation", 2)?;
                state.serialize_field("op", "Transpose")?;
                state.serialize_field("axes", axes.as_ref())?;
                state.end()
            }
            Op::Slice { offset } => {
                let mut state = serializer.serialize_struct("Operation", 2)?;
                state.serialize_field("op", "Slice")?;
                state.serialize_field("offset", offset)?;
                state.end()
            }
            Op::BitwiseAnd => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "BitwiseAnd")?;
                state.end()
            }
            Op::BitwiseOr => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "BitwiseOr")?;
                state.end()
            }
            Op::BitwiseXor => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "BitwiseXor")?;
                state.end()
            }
            Op::BitwiseNot => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "BitwiseNot")?;
                state.end()
            }
            Op::Abs => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Abs")?;
                state.end()
            }
            Op::Ceil => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Ceil")?;
                state.end()
            }
            Op::Exp => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Exp")?;
                state.end()
            }
            Op::Floor => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Floor")?;
                state.end()
            }
            Op::AsType => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "AsType")?;
                state.end()
            }
            Op::Expand => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Expand")?;
                state.end()
            }
            Op::Contiguous => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Contiguous")?;
                state.end()
            }
            Op::Sin => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Sin")?;
                state.end()
            }
            Op::Cos => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Cos")?;
                state.end()
            }
            Op::Tan => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Tan")?;
                state.end()
            }
            Op::Asin => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Asin")?;
                state.end()
            }
            Op::Acos => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Acos")?;
                state.end()
            }
            Op::Atan => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Atan")?;
                state.end()
            }
            Op::Sinh => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Sinh")?;
                state.end()
            }
            Op::Cosh => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Cosh")?;
                state.end()
            }
            Op::Tanh => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Tanh")?;
                state.end()
            }
            Op::Asinh => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Asinh")?;
                state.end()
            }
            Op::Acosh => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Acosh")?;
                state.end()
            }
            Op::Atanh => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Atanh")?;
                state.end()
            }
            Op::Null => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Null")?;
                state.end()
            }
            Op::Merge => {
                let mut state = serializer.serialize_struct("Operation", 1)?;
                state.serialize_field("op", "Merge")?;
                state.end()
            }
        }
    }
}

pub enum OpType {
    OneToMany,
    OneToOne,
    ManyToOne,
    ManyToMany,
    Opaque
}