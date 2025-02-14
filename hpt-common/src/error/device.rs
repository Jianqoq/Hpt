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

    /// CUDA driver error
    #[error("CUDA driver error: {message} at {location}")]
    #[cfg(feature = "cuda")]
    CudaDriverError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<cudarc::driver::result::DriverError>,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    #[cfg(feature = "cuda")]
    /// CUDA Cublas error
    #[error("CUDA Cublas error: {message} at {location}")]
    CudaCublasError {
        /// Error message
        message: String,
        /// Source error
        #[source]
        source: Option<cudarc::cublas::result::CublasError>,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Environment variable not set
    #[error("Environment variable {variable} not set at {location}")]
    EnvVarNotSet {
        /// Name of the environment variable that was not set
        variable: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },
}

#[cfg(feature = "cuda")]
mod impls {
    use crate::error::base::TensorError;
    use crate::error::device::DeviceError;
    use std::panic::Location;
    impl From<cudarc::driver::result::DriverError> for TensorError {
        fn from(source: cudarc::driver::result::DriverError) -> Self {
            Self::Device(DeviceError::CudaDriverError {
                message: source.to_string(),
                source: Some(source),
                location: Location::caller(),
            })
        }
    }
    impl From<cudarc::cublas::result::CublasError> for TensorError {
        fn from(source: cudarc::cublas::result::CublasError) -> Self {
            Self::Device(DeviceError::CudaCublasError {
                message: source.to_string(),
                source: Some(source),
                location: Location::caller(),
            })
        }
    }
}
