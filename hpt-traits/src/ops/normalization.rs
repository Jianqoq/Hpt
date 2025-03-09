use hpt_common::error::base::TensorError;
use hpt_common::shape::shape::Shape;
use hpt_types::into_scalar::Cast;

/// A trait contains normalization operations
pub trait NormalizationOps {
    /// The type of the output tensor
    type Output;
    /// The type of the output meta
    type OutputMeta;

    /// Applies Layer Normalization over a specified axes.
    ///
    /// ## Parameters:
    /// `normalized_shape`: shape that must match the dimension size from input tensor shape (from right to left)
    ///
    /// `gamma`: Optional scale tensor of shape `[normalized_shape]`
    ///
    /// `beta`: Optional bias tensor of shape `[normalized_shape]`
    ///
    /// `eps`: A value added to the denominator for numerical stability.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[2, 3, 4])?;
    /// let gamma = Tensor::<f32>::ones(&[4])?;
    /// let beta = Tensor::<f32>::zeros(&[4])?;
    /// let result = x.layernorm(&[4], Some(&gamma), Some(&beta), 1e-5)?;
    /// ```
    fn layernorm<S: Into<Shape>>(
        &self,
        normalized_shape: S,
        gamma: Option<&Self::Output>,
        beta: Option<&Self::Output>,
        eps: Self::OutputMeta,
    ) -> Result<Self::Output, TensorError>
    where
        usize: Cast<Self::OutputMeta>;

    /// Applies the softmax function to the input tensor along the specified dimension.
    /// The softmax function normalizes the input to a probability distribution, such that each element is in the range [0, 1] and all elements sum to 1.
    ///
    /// ## Parameters:
    /// `dim`: The dimension along which to apply the softmax.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::new(&[[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]);
    /// let result = x.softmax(1)?;
    /// ```
    fn softmax(&self, axis: i64) -> Result<Self::Output, TensorError>;

    /// Applies the log-softmax function to the input tensor along the specified dimension.
    /// The log-softmax function is equivalent to applying the logarithm to the output of the softmax function, but is more numerically stable when computed directly.
    ///
    /// ## Parameters:
    /// `dim`: The dimension along which to apply the log-softmax.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::new(&[[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]);
    /// let result = x.log_softmax(1)?;
    /// ```
    fn log_softmax(&self, axis: i64) -> Result<Self::Output, TensorError>;
}
