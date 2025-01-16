use std::panic::Location;

use thiserror::Error;

/// Errors related to autograd
#[derive(Debug, Error)]
pub enum AutogradError {
    /// Error that occurs when inplace computation is not allowed in autograd
    #[error("Inplace computation is not allowed in autograd, at {location}")]
    InplaceCompError {
        /// Location where the error occurred
        location: &'static Location<'static>,
    },
}