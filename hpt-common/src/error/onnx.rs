use std::panic::Location;

use thiserror::Error;

/// Onnx-related errors
#[derive(Debug, Error)]
pub enum OnnxError {
    /// Onnx error
    #[error("Onnx error: {message} at {location}")]
    Any {
        /// Message describing the error
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },
}

impl OnnxError {
    /// Create a new Onnx error
    #[track_caller]
    pub fn new(message: String) -> Self {
        Self::Any { message, location: Location::caller() }
    }
}
