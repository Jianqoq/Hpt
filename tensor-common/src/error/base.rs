use thiserror::Error;

use super::{device::DeviceError, kernel::KernelError, memory::MemoryError, param::ParamError, shape::ShapeError};

/// Base error type for all tensor operations
#[derive(Debug, Error)]
pub enum TensorError {
    /// Shape-related errors such as dimension mismatch, broadcasting errors
    #[error(transparent)]
    Shape(#[from] ShapeError),

    /// Device-related errors such as device not found, CUDA errors
    #[error(transparent)]
    Device(#[from] DeviceError),

    /// Memory-related errors such as memory allocation failed, invalid memory layout
    #[error(transparent)]
    Memory(#[from] MemoryError),

    /// Kernel-related errors such as kernel compilation failed, kernel execution failed
    #[error(transparent)]
    Kernel(#[from] KernelError),

    /// Parameter-related errors such as invalid function arguments
    #[error(transparent)]
    Param(#[from] ParamError),
}
