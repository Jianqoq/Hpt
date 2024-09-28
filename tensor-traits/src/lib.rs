//! this crate defines a set of traits for tensor and its operations.

#![deny(missing_docs)]

/// A module contains tensor traits
pub mod tensor;
/// A module contains tensor operations traits
pub mod ops {
    /// A module contains unary operations
    pub mod uary;
    /// A module contains binary operations
    pub mod binary;
    /// A module contains fft operations
    pub mod fft;
    /// A module contains comparison operations
    pub mod cmp;
}
/// A module contains shape manipulation operations
pub mod shape_manipulate;
/// A module contains random number generation operations
pub mod random;

pub use ops::uary::*;
pub use ops::binary::*;
pub use ops::fft::*;
pub use ops::cmp::*;
pub use shape_manipulate::*;
pub use random::*;
pub use tensor::*;
