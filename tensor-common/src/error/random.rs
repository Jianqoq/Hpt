use std::panic::Location;

use rand_distr::{
    BernoulliError, BetaError, ChiSquaredError, ExpError, GammaError, GumbelError, NormalError,
    NormalInverseGaussianError, ParetoError, PoissonError, TriangularError, WeibullError,
    ZipfError,
};
use thiserror::Error;

use super::base::TensorError;

/// Random distribution-related errors
#[derive(Debug, Error)]
pub enum RandomError {
    /// Beta distribution error
    #[error("Beta distribution error: {source} at {location}")]
    Beta {
        /// Beta distribution error
        source: BetaError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Normal distribution error
    #[error("Normal distribution error: {source} at {location}")]
    Normal {
        /// Normal distribution error
        source: NormalError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Chi-square distribution error
    #[error("Chi-square distribution error: {source} at {location}")]
    ChiSquare {
        /// Chi-square distribution error
        source: ChiSquaredError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Exponential distribution error
    #[error("Exponential distribution error: {source} at {location}")]
    Exp {
        /// Exponential distribution error
        source: ExpError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Gamma distribution error
    #[error("Gamma distribution error: {source} at {location}")]
    Gamma {
        /// Gamma distribution error
        source: GammaError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Gumbel distribution error
    #[error("Gumbel distribution error: {source} at {location}")]
    Gumbel {
        /// Gumbel distribution error
        source: GumbelError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Inverse Gaussian distribution error
    #[error("Inverse Gaussian distribution error: {source} at {location}")]
    NormalInverseGaussian {
        /// Inverse Gaussian distribution error
        source: NormalInverseGaussianError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Pareto distribution error
    #[error("Pareto distribution error: {source} at {location}")]
    Pareto {
        /// Pareto distribution error
        source: ParetoError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Poisson distribution error
    #[error("Poisson distribution error: {source} at {location}")]
    Poisson {
        /// Poisson distribution error
        source: PoissonError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Weibull distribution error
    #[error("Weibull distribution error: {source} at {location}")]
    Weibull {
        /// Weibull distribution error
        source: WeibullError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Zipf distribution error
    #[error("Zipf distribution error: {source} at {location}")]
    Zipf {
        /// Zipf distribution error
        source: ZipfError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Triangular distribution error
    #[error("Triangular distribution error: {source} at {location}")]
    Triangular {
        /// Triangular distribution error
        source: TriangularError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
    /// Bernoulli distribution error
    #[error("Bernoulli distribution error: {source} at {location}")]
    Bernoulli {
        /// Bernoulli distribution error
        source: BernoulliError,
        /// Location where error occurred
        location: &'static Location<'static>,
    },
}

impl From<BetaError> for TensorError {
    #[track_caller]
    fn from(source: BetaError) -> Self {
        Self::Random(RandomError::Beta {
            source,
            location: Location::caller(),
        })
    }
}

impl From<NormalError> for TensorError {
    #[track_caller]
    fn from(source: NormalError) -> Self {
        Self::Random(RandomError::Normal {
            source,
            location: Location::caller(),
        })
    }
}

impl From<ChiSquaredError> for TensorError {
    #[track_caller]
    fn from(source: ChiSquaredError) -> Self {
        Self::Random(RandomError::ChiSquare {
            source,
            location: Location::caller(),
        })
    }
}

impl From<ExpError> for TensorError {
    #[track_caller]
    fn from(source: ExpError) -> Self {
        Self::Random(RandomError::Exp {
            source,
            location: Location::caller(),
        })
    }
}

impl From<GammaError> for TensorError {
    #[track_caller]
    fn from(source: GammaError) -> Self {
        Self::Random(RandomError::Gamma {
            source,
            location: Location::caller(),
        })
    }
}

impl From<GumbelError> for TensorError {
    #[track_caller]
    fn from(source: GumbelError) -> Self {
        Self::Random(RandomError::Gumbel {
            source,
            location: Location::caller(),
        })
    }
}

impl From<NormalInverseGaussianError> for TensorError {
    #[track_caller]
    fn from(source: NormalInverseGaussianError) -> Self {
        Self::Random(RandomError::NormalInverseGaussian {
            source,
            location: Location::caller(),
        })
    }
}

impl From<ParetoError> for TensorError {
    #[track_caller]
    fn from(source: ParetoError) -> Self {
        Self::Random(RandomError::Pareto {
            source,
            location: Location::caller(),
        })
    }
}

impl From<PoissonError> for TensorError {
    #[track_caller]
    fn from(source: PoissonError) -> Self {
        Self::Random(RandomError::Poisson {
            source,
            location: Location::caller(),
        })
    }
}

impl From<WeibullError> for TensorError {
    #[track_caller]
    fn from(source: WeibullError) -> Self {
        Self::Random(RandomError::Weibull {
            source,
            location: Location::caller(),
        })
    }
}

impl From<ZipfError> for TensorError {
    #[track_caller]
    fn from(source: ZipfError) -> Self {
        Self::Random(RandomError::Zipf {
            source,
            location: Location::caller(),
        })
    }
}

impl From<TriangularError> for TensorError {
    #[track_caller]
    fn from(source: TriangularError) -> Self {
        Self::Random(RandomError::Triangular {
            source,
            location: Location::caller(),
        })
    }
}

impl From<BernoulliError> for TensorError {
    #[track_caller]
    fn from(source: BernoulliError) -> Self {
        Self::Random(RandomError::Bernoulli {
            source,
            location: Location::caller(),
        })
    }
}
