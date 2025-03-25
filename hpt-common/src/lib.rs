//! This crate contains common utilities for tensor manipulation.

#![deny(missing_docs)]

/// A module defines n dimensional strides
pub mod strides {
    /// strides struct definition
    pub mod strides;
    /// A module contains strides utilities
    pub mod strides_utils;
}
/// A module defines n dimensional shape
pub mod shape {
    /// shape struct definition
    pub mod shape;
    /// A module contains shape utilities
    pub mod shape_utils;
}

/// A module defines layout
pub mod layout {
    /// layout struct definition
    pub mod layout;
    /// A module contains layout utilities
    pub mod layout_utils;
}

/// A module defines axis
pub mod axis {
    /// axis struct definition
    pub mod axis;
}

/// A module defines utilities
pub mod utils {
    /// A module defines conv parameters
    pub mod conv_algos;
    /// A module defines pointer utilities
    pub mod pointer;
    /// this module defines simd vector reference, this force the user to use write unaligned and read unaligned when they use simd iterator
    pub mod simd_ref;
    /// A module defines tensordot function arguments
    pub mod tensordot_args;
}

/// A module defines loop progress update
pub mod prg_update;
/// A module defines slice utilities
pub mod slice;

pub mod error;

pub use utils::pointer::Pointer;
