use rand_distr::BetaError;
use thiserror::Error;

/// Random distribution-related errors
#[derive(Debug, Error)]
pub enum RandomError {
    /// Beta distribution error
    #[error("Beta distribution error: {0}")]
    Beta(#[from] BetaError),
}

