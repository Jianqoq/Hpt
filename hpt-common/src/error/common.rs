use std::panic::Location;

use thiserror::Error;

/// Errors that can occur during kernel operations (compilation, execution)
#[derive(Debug, Error)]
pub enum CommonError {
    /// Error that occurs when lock failed
    #[error("Lock failed: {message} at {location}")]
    LockFailed {
        /// Message describing the lock failure
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },
}
