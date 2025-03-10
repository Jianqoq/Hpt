use hpt_common::{error::base::TensorError, shape::shape::Shape};

/// A trait defines empty function for Tensor that will allocate memory on CPU.
pub trait CPUTensorCreator {
    /// the output type of the creator
    type Output;
    /// the meta type of the tensor
    type Meta;

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

/// A trait defines conversion to DataLoader
pub trait ToDataLoader {
    /// the output type of the conversion
    type Output;
    /// convert to DataLoader
    fn to_dataloader(self) -> Self::Output;
}
