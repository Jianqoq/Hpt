use hpt_common::error::base::TensorError;
use hpt_types::{into_scalar::Cast, type_promote::NormalOut};

/// A trait contains regularization operations
pub trait RegularizationOps {
    /// The type of the output tensor
    type Output;
    /// The type of the output meta
    type OutputMeta;

    /// Randomly zeroes some of the elements of the input tensor with probability rate using samples from a Bernoulli distribution. Each element is zeroed independently.
    ///
    /// ## Parameters:
    /// `rate`: Probability of an element to be zeroed. The value must be between 0 and 1.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::ones(&[3, 4])?;
    /// let dropped = x.dropout(0.5)?;
    /// ```
    fn dropout(&self, rate: f64) -> Result<Self::Output, TensorError>
    where
        f64: Cast<Self::OutputMeta>,
        bool: Cast<Self::OutputMeta>,
        Self::OutputMeta: NormalOut<bool, Output = Self::OutputMeta>;

    /// Applies the shrinkage function to the input tensor. The shrinkage function is a soft thresholding operator commonly used in signal processing and optimization algorithms, defined as:
    /// `sign(x - bias) * max(abs(x - bias) - lambda, 0)`
    ///
    /// ## Parameters:
    /// `bias`: Bias value to subtract from each element before applying shrinkage.
    ///
    /// `lambda`: Threshold parameter controlling the amount of shrinkage.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::new(&[[-3.0, -1.0, 0.0, 2.0, 5.0]]);
    /// let result = x.shrinkage(0.0, 1.5)?; // [[-1.5, 0.0, 0.0, 0.5, 3.5]]
    /// ```
    fn shrinkage(
        &self,
        bias: Self::OutputMeta,
        lambda: Self::OutputMeta,
    ) -> Result<Self::Output, TensorError>;
}
