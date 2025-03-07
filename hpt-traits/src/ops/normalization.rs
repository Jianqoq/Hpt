use hpt_common::error::base::TensorError;
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};

/// A trait contains normalization operations
pub trait NormalizationOps {
    /// The type of the output tensor
    type Output;
    /// The type of the inplace output tensor
    type InplaceOutput;
    /// The type of the output meta
    type OutputMeta;

    /// Applies Layer Normalization over a mini-batch of inputs.
    ///
    /// Layer Normalization normalizes the input across the normalized_shape dimensions,
    /// and optionally applies an affine transformation (scale and shift) using gamma and beta parameters.
    fn layernorm<S>(
        &self,
        normalized_shape: S,
        gamma: Option<&Self::Output>,
        beta: Option<&Self::Output>,
        eps: Self::OutputMeta
    ) -> Result<Self::Output, TensorError>;

    /// Applies the Softmax function to the input tensor.
    ///
    /// The Softmax function computes the exponential of the input tensor along a specified axis and normalizes the result
    /// to ensure the sum of the output values equals 1.
    fn softmax(&self, axis: i64) -> Result<Self::Output, TensorError>;

    /// Applies the LogSoftmax function to the input tensor.
    ///
    /// The LogSoftmax function computes the natural logarithm of the Softmax function applied to the input tensor.
    fn log_softmax(&self, axis: i64) -> Result<Self::Output, TensorError>;

    /// Gather the tensor
    // fn gather(&self, indices: &Self::IndexOutput, axis: i64) -> Result<Self::Output, TensorError>;
    /// Dropout the tensor
    fn dropout(&self, rate: f64) -> Result<Self::Output, TensorError>
    where
        f64: Cast<Self::OutputMeta>,
        Self::OutputMeta: NormalOut<bool, Output = Self::OutputMeta>;
}
