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
}

impl ParamError {
    /// Check if the trim parameter is valid
    pub fn check_trim(value: &str) -> Result<(), Self> {
        if !(value == "fb" || value == "f" || value == "b") {
            return Err(ParamError::InvalidTrimParam{
                value: value.to_string(),
                location: Location::caller(),
            }
            .into());
        }
        Ok(())
    }
}

