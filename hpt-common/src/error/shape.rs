use std::panic::Location;

use thiserror::Error;

use crate::{shape::shape::Shape, strides::strides::Strides};

/// Errors related to tensor shapes and dimensions
#[derive(Debug, Error)]
pub enum ShapeError {
    /// Error that occurs when the size of two tensors does not match
    #[error("Size mismatch: expected {expected}, got {actual} at {location}")]
    SizeMismatch {
        /// Expected size
        expected: i64,
        /// Actual size
        actual: i64,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the dimension of a tensor is invalid
    #[error("Invalid dimension: {message} at {location}")]
    InvalidDimension {
        /// Message describing the invalid dimension
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when broadcasting fails
    #[error("Broadcasting error: {message} at {location}")]
    BroadcastError {
        /// Message describing the broadcasting failure
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the shape of two tensors does not match for matrix multiplication
    #[error("Matrix multiplication shape mismatch: lhs shape {lhs:?}, rhs shape {rhs:?}, expected rhs shape [{expected}, N] at {location}")]
    MatmulMismatch {
        /// Left-hand side shape
        lhs: Shape,
        /// Right-hand side shape
        rhs: Shape,
        /// Expected shape for the right-hand side
        expected: i64,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the dimension of two tensors does not match
    #[error("Dimension mismatch: expected {expected}, got {actual} at {location}")]
    DimMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the dimension is out of range
    #[error("Dimension out of range: expected in {expected:?}, got {actual} at {location}")]
    DimOutOfRange {
        /// Expected range
        expected: std::ops::Range<i64>,
        /// Actual dimension
        actual: i64,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when geomspace parameters are invalid
    #[error("Geomspace error: start {start} and end {end} must have the same sign at {location}")]
    GeomSpaceError {
        /// Start value
        start: f64,
        /// End value
        end: f64,
        /// Location where error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when concat dimensions don't match
    #[error("Concat dimension mismatch: expected {expected} but got {actual} at {location}")]
    ConcatDimMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
        /// Location where error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the number of dimensions of a tensor is less than the expected value
    #[error("Ndim not enough: expected greater than {expected}, got {actual} at {location}")]
    NdimNotEnough {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
        /// Location where error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the axis is not 1
    #[error("Squeeze error: axis {axis} is not 1, shape {shape}, at {location}")]
    SqueezeError {
        /// Axis that is not 1
        axis: usize,
        /// Shape of the tensor
        shape: Shape,
        /// Location where error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the tensor is not contiguous
    #[error("{message}Tensor is not contiguous, got shape {shape:?}, strides {strides:?}, at {location}")]
    ContiguousError {
        /// message
        message: String,
        /// Shape of the tensor
        shape: Shape,
        /// Strides of the tensor
        strides: Strides,
        /// Location where error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the input shape is invalid for conv2d
    #[error("Conv error: {message} at {location}")]
    ConvError {
        /// Message describing the invalid input shape
        message: String,
        /// Location where error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the topk operation is invalid
    #[error("Topk error: {message} at {location}")]
    TopkError {
        /// Message describing the invalid topk operation
        message: String,
        /// Location where error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the inplace reshape is invalid
    #[error("Inplace reshape error: {message} at {location}")]
    InplaceReshapeError {
        /// Message describing the invalid inplace reshape
        message: String,
        /// Location where error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when the dimention to expand is not 1
    #[error("Expand error: dimention {old_dim} is not 1, at {location}")]
    ExpandError {
        /// Old dimention
        old_dim: i64,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
}

impl ShapeError {
    /// Check if the shapes of two tensors match for matrix multiplication
    #[track_caller]
    pub fn check_matmul(lhs: &Shape, rhs: &Shape) -> Result<(), Self> {
        let lhs_last = *lhs.last().expect("lhs shape is empty");
        let rhs_last_sec = rhs[rhs.len() - 2];
        if lhs_last != rhs_last_sec {
            return Err(Self::MatmulMismatch {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                expected: lhs_last,
                location: Location::caller(),
            });
        }
        Ok(())
    }

    /// Check if the dimensions of two tensors match
    #[track_caller]
    pub fn check_dim(expected: usize, actual: usize) -> Result<(), Self> {
        if expected != actual {
            return Err(Self::DimMismatch {
                expected,
                actual,
                location: Location::caller(),
            });
        }
        Ok(())
    }

    /// Check if the number of dimensions of a tensor is greater than the expected value
    #[track_caller]
    pub fn check_ndim_enough(expected: usize, actual: usize) -> Result<(), Self> {
        if expected > actual {
            return Err(Self::NdimNotEnough {
                expected,
                actual,
                location: Location::caller(),
            });
        }
        Ok(())
    }

    /// Check if the tensor is contiguous
    #[track_caller]
    pub fn check_contiguous(
        msg: String,
        layout: &crate::layout::layout::Layout,
    ) -> Result<(), Self> {
        if !layout.is_contiguous() {
            return Err(Self::ContiguousError {
                message: msg,
                shape: layout.shape().clone(),
                strides: layout.strides().clone(),
                location: Location::caller(),
            });
        }
        Ok(())
    }

    /// Check if the size of two tensors match
    #[track_caller]
    pub fn check_size_match(expected: i64, actual: i64) -> Result<(), Self> {
        if expected != actual {
            return Err(Self::SizeMismatch {
                expected,
                actual,
                location: Location::caller(),
            });
        }
        Ok(())
    }

    /// Check if the output layout is valid for computation with inplace operation
    #[track_caller]
    pub fn check_inplace_out_layout_valid(
        out_shape: &Shape,
        inplace_layout: &crate::layout::layout::Layout,
    ) -> Result<(), Self> {
        Self::check_size_match(out_shape.size(), inplace_layout.size())?;
        Self::check_contiguous(
            "Method with out Tensor requires out Tensor to be contiguous. ".to_string(),
            inplace_layout,
        )?;
        Ok(())
    }

    /// Check if the index is out of range
    #[track_caller]
    pub fn check_index_out_of_range(index: i64, dim: i64) -> Result<(), Self> {
        if index >= dim || index < 0 {
            return Err(Self::DimOutOfRange {
                expected: 0..dim,
                actual: index,
                location: Location::caller(),
            });
        }

        Ok(())
    }
}
