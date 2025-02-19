use hpt_common::{error::base::TensorError, shape::shape::Shape};

pub(crate) fn create_file(path: std::path::PathBuf, ext: &str) -> std::io::Result<std::fs::File> {
    if let Some(extension) = path.extension() {
        if extension == ext {
            std::fs::File::create(path)
        } else {
            std::fs::File::create(format!("{}.{ext}", path.to_str().unwrap()))
        }
    } else {
        std::fs::File::create(format!("{}.{ext}", path.to_str().unwrap()))
    }
}

/// A trait defines empty function for Tensor that will allocate memory on CPU.
pub trait CPUTensorCreator<T> {
    /// the output type of the creator
    type Output;

    /// Creates a tensor with uninitialized elements of the specified shape.
    ///
    /// This function allocates memory for a tensor in CPU of the given shape, but the values are uninitialized, meaning they may contain random data.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the tensor. The type `S` must implement `Into<Shape>`.
    ///
    /// # Returns
    ///
    /// * A tensor with the specified shape, but with uninitialized data.
    ///
    /// # Panics
    ///
    /// * This function may panic if the requested shape is invalid or too large for available memory.
    #[track_caller]
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>;
}
