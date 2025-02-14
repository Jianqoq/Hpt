use std::panic::Location;

use thiserror::Error;

/// Errors related to memory operations such as allocation, layout validation
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Error that occurs when memory allocation fails
    #[error("Failed to allocate {size} bytes on {device}{id} at {location}")]
    AllocationFailed {
        /// Name of the device where the allocation failed
        device: String,
        /// ID of the device where the allocation failed
        id: usize,
        /// Size of the memory that was attempted to be allocated
        size: usize,
        /// Source of the error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when an invalid memory layout is detected
    #[error("Invalid memory layout: {message} at {location}")]
    InvalidLayout {
        /// Message describing the invalid layout
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },
    /// used when the reference count overflow
    #[error("reference count overflow for device {device}{id}, at {location}")]
    ReferenceCountOverflow {
        /// Name of the device where the allocation failed
        device: String,
        /// ID of the device where the allocation failed
        id: usize,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
}