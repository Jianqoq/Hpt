//! this crate defines a set of traits for tensor and its operations.

#![deny(missing_docs)]

/// A module contains tensor traits
pub mod tensor;
/// A module contains tensor operations traits
pub mod ops {
    /// A module contains advance operations
    pub mod advance;
    /// A module contains binary operations
    pub mod binary;
    /// A module contains comparison operations
    pub mod cmp;
    /// A module contains conv operations
    pub mod conv;
    /// A module contains cumulative operations
    pub mod cumulative;
    /// A module contains fft operations
    pub mod fft;
    /// A module contains pooling operations
    pub mod pooling;
    /// A module contains slice operations
    pub mod slice;
    /// A module contains unary operations
    pub mod uary;
    /// A module contains window operations
    pub mod windows;
    /// A module contains normalization operations
    pub mod normalization;
}
/// A module contains random number generation operations
pub mod random;
/// A module contains shape manipulation operations
pub mod shape_manipulate;

pub use ops::advance::*;
pub use ops::binary::*;
pub use ops::cmp::*;
pub use ops::conv::*;
pub use ops::cumulative::*;
pub use ops::fft::*;
pub use ops::pooling::*;
pub use ops::slice::*;
pub use ops::uary::*;
pub use ops::windows::*;
pub use ops::normalization::*;
pub use random::*;
pub use shape_manipulate::*;
pub use tensor::*;
