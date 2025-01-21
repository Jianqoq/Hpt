//! Error handling for tensor operations
//!
//! This module contains various error types used throughout the tensor library,
//! organized by their domains (shape, device, memory, kernel, etc.)

/// Base error types and common functionality
pub mod base;
/// Device-related errors (GPU, CPU, etc.)
pub mod device;
/// Kernel-related errors (CUDA, etc.)
pub mod kernel;
/// Memory allocation and management errors
pub mod memory;
/// Shape-related errors (dimension mismatch, broadcasting, etc.)
pub mod shape;
/// Parameter-related errors (function arguments, etc.)
pub mod param;
/// Autograd-related errors (gradient computation, etc.)
pub mod autograd;
/// Random distribution-related errors
pub mod random;
