use std::panic::Location;

use thiserror::Error;

/// Errors that can occur during kernel operations (compilation, execution)
#[derive(Debug, Error)]
pub enum KernelError {
    /// Error that occurs when kernel compilation fails
    #[error("Kernel compilation failed: {message} at {location}")]
    CompilationFailed {
        /// Message describing the compilation failure
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when kernel execution fails
    #[error("Kernel execution failed: {message} at {location}")]
    ExecutionFailed {
        /// Message describing the execution failure
        message: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when CUDA kernel region info is not found
    #[error("CUDA kernel region info not found for module: {module}, func_name: {func_name}")]
    CudaKernelRegInfoNotFound {
        /// Module name
        module: String,
        /// Function name
        func_name: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },

    /// Error that occurs when CUDA kernel meta data is not found
    #[error(
        "CUDA kernel meta data not found for module: {module}, func_name: {func_name}, cap: {cap}"
    )]
    CudaKernelMetaNotFound {
        /// cap
        cap: usize,
        /// Module name
        module: String,
        /// Function name
        func_name: String,
        /// Location where the error occurred
        location: &'static Location<'static>,
    },
}

#[cfg(feature = "cuda")]
mod impls {
    use crate::error::base::TensorError;
    use crate::error::kernel::KernelError;
    use std::panic::Location;
    impl From<cudarc::nvrtc::CompileError> for TensorError {
        fn from(source: cudarc::nvrtc::CompileError) -> Self {
            Self::Kernel(KernelError::CompilationFailed {
                message: source.to_string(),
                location: Location::caller(),
            })
        }
    }
}
