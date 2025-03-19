use std::panic::Location;

use thiserror::Error;

/// Parameter-related errors such as invalid function arguments
#[derive(Debug, Error)]
pub enum ParamError {
    /// Error that occurs when trim parameter is invalid
    #[error("Invalid trim parameter: must be one of 'fb', 'f', 'b', got {value} at {location}")]
    InvalidTrimParam {
        /// Invalid trim value
        value: String,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Error that occurs when the axis is duplicated
    #[error("Axis {axis} is duplicated at {location}")]
    AxisDuplicated {
        /// Duplicated axis
        axis: i64,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Error that occurs when FFT norm parameter is invalid
    #[error("Invalid FFT norm parameter: must be one of 'backward', 'forward', 'ortho', got {value} at {location}")]
    InvalidFFTNormParam {
        /// Invalid FFT norm value
        value: String,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
}

impl ParamError {
    /// Check if the trim parameter is valid
    pub fn check_trim(value: &str) -> Result<(), Self> {
        if !(value == "fb" || value == "f" || value == "b") {
            return Err(ParamError::InvalidTrimParam {
                value: value.to_string(),
                location: Location::caller(),
            }
            .into());
        }
        Ok(())
    }
}
