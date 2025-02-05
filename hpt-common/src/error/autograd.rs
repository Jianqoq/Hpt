use std::panic::Location;

use thiserror::Error;

/// Errors related to autograd
#[derive(Debug, Error)]
pub enum AutogradError {
    /// Error that occurs when inplace computation is not allowed in autograd
    #[error("Inplace computation {op} is not allowed in autograd, at {location}")]
    InplaceCompError {
        /// Operation name
        op: &'static str,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },
    /// Error that occurs when the operation is not supported in autograd
    #[error("Operation {op} is not supported in autograd, at {location}")]
    UnsupportOpError {
        /// Operation name
        op: &'static str,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },
}
