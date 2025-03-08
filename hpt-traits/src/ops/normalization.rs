use hpt_common::error::base::TensorError;
use hpt_common::shape::shape::Shape;
use hpt_types::into_scalar::Cast;

/// A trait contains normalization operations
pub trait NormalizationOps {
    /// The type of the output tensor
    type Output;
    /// The type of the output meta
    type OutputMeta;

    /// Applies Layer Normalization over a mini-batch of inputs.
    ///
    /// Layer Normalization normalizes the input across the normalized_shape dimensions,
    /// and optionally applies an affine transformation (scale and shift) using gamma and beta parameters.
    fn layernorm<S: Into<Shape>>(
        &self,
        normalized_shape: S,
        gamma: Option<&Self::Output>,
        beta: Option<&Self::Output>,
        eps: Self::OutputMeta,
    ) -> Result<Self::Output, TensorError>
    where
        usize: Cast<Self::OutputMeta>;

    /// Applies the Softmax function to the input tensor.
    ///
    /// The Softmax function computes the exponential of the input tensor along a specified axis and normalizes the result
    /// to ensure the sum of the output values equals 1.
    fn softmax(&self, axis: i64) -> Result<Self::Output, TensorError>;

    /// Applies the LogSoftmax function to the input tensor.
    ///
    /// The LogSoftmax function computes the natural logarithm of the Softmax function applied to the input tensor.
    fn log_softmax(&self, axis: i64) -> Result<Self::Output, TensorError>;
}
