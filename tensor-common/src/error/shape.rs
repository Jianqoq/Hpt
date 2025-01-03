use std::panic::Location;

use thiserror::Error;

use crate::shape::Shape;

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
        lhs: [i64; 2],
        /// Right-hand side shape
        rhs: [i64; 2],
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
}

impl ShapeError {
    /// Check if the shapes of two tensors match for matrix multiplication
    pub fn check_matmul(lhs: &Shape, rhs: &Shape) -> Result<(), Self> {
        unimplemented!()
    }

    /// Check if the shapes of two tensors match for broadcasting
    pub fn check_broadcast(lhs: &Shape, rhs: &Shape) -> Result<(), Self> {
        unimplemented!()
    }

    /// Check if the dimensions of two tensors match
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
}
