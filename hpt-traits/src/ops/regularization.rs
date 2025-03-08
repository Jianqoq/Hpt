use hpt_common::error::base::TensorError;
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};

/// A trait contains regularization operations
pub trait RegularizationOps {
    /// The type of the output tensor
    type Output;
    /// The type of the output meta
    type OutputMeta;

    /// Applies dropout to the input tensor.
    ///
    /// This function randomly zeroes some of the elements of the input tensor with probability `rate`
    /// using samples from a Bernoulli distribution. Each element is zeroed independently.
    fn dropout(&self, rate: f64) -> Result<Self::Output, TensorError>
    where
        f64: Cast<Self::OutputMeta>,
        bool: Cast<Self::OutputMeta>,
        Self::OutputMeta: NormalOut<bool, Output = Self::OutputMeta>;

    /// Applies the shrinkage function to the input tensor. The shrinkage function is a soft thresholding operator commonly used in signal processing and optimization algorithms, defined as:
    ///
    /// sign(x - bias) * max(abs(x - bias) - lambda, 0)
    fn shrinkage(
        &self,
        bias: Self::OutputMeta,
        lambda: Self::OutputMeta,
    ) -> Result<Self::Output, TensorError>;
}
