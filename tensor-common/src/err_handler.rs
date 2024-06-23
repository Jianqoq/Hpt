use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{shape::Shape, strides::Strides};

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub enum ErrHandler {
    AxisMismatched(String),
    SameAxisError(String),
    SizeMismatched(String),
    MatmulShapeMismatched(String),
    NdimMismatched(String),
    AllocRequired(String),
    IndexOutOfRange(String),
    IndexRepeated(String),
    ExpandDimError(String),
    InplaceReshapeError(String),
    BroadcastError(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum ErrHandlerType {
    SizeMismatched,
    NdimMismatched,
    AllocRequired,
    IndexOutOfRange,
    IndexRepeated,
    ExpandDimError,
    AxisMismatched,
    SameAxisError,
    InplaceReshapeError,
    BroadcastError,
}

impl ErrHandler {
    pub fn check_size_match(lhs: &[i64], rhs: &[i64]) -> anyhow::Result<(), ErrHandler> {
        if lhs.iter().product::<i64>() != rhs.iter().product::<i64>() {
            Err(ErrHandler::SizeMismatched(format!(
                "size mismatched, lhs: {}, rhs: {}.",
                lhs.iter().product::<i64>(),
                rhs.iter().product::<i64>()
            )))
        } else {
            Ok(())
        }
    }

    pub fn check_ndim_match(lhs: usize, rhs: usize) -> anyhow::Result<(), ErrHandler> {
        if lhs != rhs {
            Err(ErrHandler::NdimMismatched(format!(
                "ndim mismatched, lhs: {}, rhs: {}.",
                lhs, rhs
            )))
        } else {
            Ok(())
        }
    }

    pub fn check_axes_in_range<'a, Iter: Iterator<Item = i64>>(ndim: usize, axes: Iter) -> anyhow::Result<(), ErrHandler> {
        let mut set = HashSet::new();
        let ndim = ndim as i64;
        for (idx, axis) in axes.enumerate() {
            if axis < 0 {
                let val = axis % ndim;
                let val = if val < 0 { val + ndim as i64 } else { val };
                if !set.insert(val) {
                    return Err(ErrHandler::IndexRepeated(format!(
                        "axes[{}] {} repeated.",
                        idx, val
                    )));
                }
            } else {
                if axis >= ndim  {
                    return Err(ErrHandler::IndexOutOfRange(format!(
                        "axes[{}]: {} out of range(should be in range {}..{}).",
                        idx, axis, 0, ndim
                    )));
                }
                if !set.insert(axis) {
                    return Err(ErrHandler::IndexRepeated(format!(
                        "axes[{}] {} repeated.",
                        idx, axis
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn check_index_in_range(ndim: usize, axes: usize) -> anyhow::Result<(), ErrHandler> {
        if axes >= ndim {
            Err(ErrHandler::IndexOutOfRange(format!(
                "axes: {} out of range(should be in range {}..{}).",
                axes, 0, ndim
            )))
        } else {
            Ok(())
        }
    }

    pub fn check_axis_same(lhs: i64, rhs: i64) -> anyhow::Result<(), ErrHandler> {
        if lhs == rhs {
            Err(ErrHandler::SameAxisError(format!(
                "axis should not be the same, lhs: {}, rhs: {}.",
                lhs, rhs
            )))
        } else {
            Ok(())
        }
    }

    pub fn check_matmul_shape(lhs: &[i64], rhs: &[i64]) -> anyhow::Result<(), ErrHandler> {
        match (lhs.get(1), rhs.get(0)) {
            (Some(lhs), Some(rhs)) => {
                if lhs != rhs {
                    return Err(ErrHandler::MatmulShapeMismatched(format!(
                        "lhs shape[1] should be equal to rhs shape[0], got shape {:?} and {:?}",
                        lhs, rhs
                    )));
                } else {
                    return Ok(());
                }
            }
            (None, Some(rhs)) => {
                return Err(ErrHandler::IndexOutOfRange(format!(
                    "lhs shape should have at least 2 dimensions, got shape {:?}",
                    rhs
                )))
            }
            (Some(lhs), None) => {
                return Err(ErrHandler::IndexOutOfRange(format!(
                    "rhs shape should have at least 2 dimensions, got shape {:?}",
                    lhs
                )))
            }
            (None, None) => {
                return Err(ErrHandler::IndexOutOfRange(format!(
                    "lhs and rhs shape should have at least 2 dimensions, got shape {:?} and {:?}",
                    lhs, rhs
                )))
            }
        }
    }

    pub fn check_expand_dims(
        shape: &Shape,
        expand_shape: &Shape,
    ) -> anyhow::Result<(), ErrHandler> {
        for (x, (idx, y)) in expand_shape
            .iter()
            .rev()
            .zip(shape.iter().rev().enumerate())
        {
            if x != y && y != &1 {
                return Err(ErrHandler::ExpandDimError(format!(
                        "input shape {:?} can not be expanded to {:?}, please make sure input shape[{}]: {} equals 1 if you want to expand",
                        shape,
                        expand_shape,
                        idx,
                        shape[idx]
                    )));
            }
        }
        Ok(())
    }

    pub fn raise_requires_allocation_when_use_iterator(
        shape: &Shape,
        strides: &Strides,
    ) -> anyhow::Result<(), ErrHandler> {
        Err(ErrHandler::AllocRequired(format!(
            "Detected allocation required when using iterator, shape: {:?}, strides: {:?}",
            shape, strides
        )))
    }

    pub fn raise_broadcast_error<T>(lhs: &Shape, rhs: &Shape) -> anyhow::Result<T, ErrHandler> {
        Err(ErrHandler::BroadcastError(format!(
            "Broadcast error, lhs: {:?}, rhs: {:?}",
            lhs, rhs
        )))
    }

    pub fn to_type(&self) -> ErrHandlerType {
        match self {
            ErrHandler::SizeMismatched(_) => ErrHandlerType::SizeMismatched,
            ErrHandler::NdimMismatched(_) => ErrHandlerType::NdimMismatched,
            ErrHandler::AllocRequired(_) => ErrHandlerType::AllocRequired,
            ErrHandler::IndexOutOfRange(_) => ErrHandlerType::IndexOutOfRange,
            ErrHandler::IndexRepeated(_) => ErrHandlerType::IndexRepeated,
            ErrHandler::ExpandDimError(_) => ErrHandlerType::ExpandDimError,
            ErrHandler::AxisMismatched(_) => ErrHandlerType::AxisMismatched,
            ErrHandler::SameAxisError(_) => ErrHandlerType::SameAxisError,
            ErrHandler::InplaceReshapeError(_) => ErrHandlerType::SizeMismatched,
            ErrHandler::BroadcastError(_) => ErrHandlerType::BroadcastError,
            ErrHandler::MatmulShapeMismatched(_) => ErrHandlerType::SizeMismatched,
        }
    }
}

impl std::fmt::Display for ErrHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrHandler::SizeMismatched(msg) => write!(f, "{}", msg),
            ErrHandler::AllocRequired(msg) => write!(f, "{}", msg),
            ErrHandler::IndexOutOfRange(msg) => write!(f, "{}", msg),
            ErrHandler::IndexRepeated(msg) => write!(f, "{}", msg),
            ErrHandler::ExpandDimError(msg) => write!(f, "{}", msg),
            ErrHandler::NdimMismatched(msg) => write!(f, "{}", msg),
            ErrHandler::AxisMismatched(msg) => write!(f, "{}", msg),
            ErrHandler::SameAxisError(msg) => write!(f, "{}", msg),
            ErrHandler::InplaceReshapeError(msg) => write!(f, "{}", msg),
            ErrHandler::BroadcastError(msg) => write!(f, "{}", msg),
            ErrHandler::MatmulShapeMismatched(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::fmt::Display for ErrHandlerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrHandlerType::SizeMismatched => write!(f, "SizeMismatched"),
            ErrHandlerType::AllocRequired => write!(f, "AllocRequired"),
            ErrHandlerType::IndexOutOfRange => write!(f, "IndexOutOfRange"),
            ErrHandlerType::IndexRepeated => write!(f, "IndexRepeated"),
            ErrHandlerType::ExpandDimError => write!(f, "ExpandDimError"),
            ErrHandlerType::NdimMismatched => write!(f, "NdimMismatched"),
            ErrHandlerType::AxisMismatched => write!(f, "AxisMismatched"),
            ErrHandlerType::SameAxisError => write!(f, "SameAxisError"),
            ErrHandlerType::InplaceReshapeError => write!(f, "InplaceReshapeError"),
            ErrHandlerType::BroadcastError => write!(f, "BroadcastError"),
        }
    }
}

impl std::error::Error for ErrHandler {}
