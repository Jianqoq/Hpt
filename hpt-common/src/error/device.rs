use std::panic::Location;

use thiserror::Error;

/// Device-related errors such as device not found, CUDA errors
#[derive(Debug, Error)]
pub enum DeviceError {
    /// Device not found
    #[error("Device {device} not found at {location}")]
    NotFound {
        /// Name or ID of the device that was not found
        device: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// CUDA error
    #[error("CUDA error: {message} at {location}")]
    #[cfg(feature = "cuda")]
    CudaError {
        message: String,
        #[source]
        source: Option<cudarc::driver::result::DriverError>,
        location: &'static Location<'static>,
    },
}
