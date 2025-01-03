use std::panic::Location;

use thiserror::Error;

/// Errors that can occur during kernel operations (compilation, execution)
#[derive(Debug, Error)]
pub enum KernelError {
    /// Error that occurs when kernel compilation fails
    #[error("Kernel compilation failed: {message} at {location}")]
    CompilationFailed {
        /// Message describing the compilation failure
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when kernel execution fails
    #[error("Kernel execution failed: {message} at {location}")]
    ExecutionFailed {
        /// Message describing the execution failure
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    }
}