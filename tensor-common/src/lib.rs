//! This crate contains common utilities for tensor manipulation.

#![deny(missing_docs)]

/// A module defines n dimensional strides
pub mod strides;
/// A module defines n dimensional shape
pub mod shape;
/// A module contains layout utilities
pub mod layout;
/// A module contains shape utilities
pub mod shape_utils;
/// A module contains strides utilities
pub mod strides_utils;
/// A module defines errors
pub mod err_handler;
/// A module defines pointer utilities
pub mod pointer;
/// A module defines axis utilities
pub mod axis;
/// A module defines slice utilities
pub mod slice;
/// A module defines tensordot function arguments
pub mod tensordot_args;
/// A module defines simd vector reference, this force the user to use write unaligned and read unaligned when they use simd iterator
pub mod simd_ref;