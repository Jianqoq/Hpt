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

    /// Error that occurs when trying to forget a tensor that is still in use
    #[error("Cannot forget tensor: {msg}")]
    CantForgetTensor {
        /// Message describing the error
        msg: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when dtype mismatch
    #[error("Dtype mismatch: {message} at {location}")]
    DtypeMismatch {
        /// Message describing the error
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },
}
