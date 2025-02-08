use hpt_common::error::base::TensorError;

use crate::CommonBounds;

/// trait for conv operations
pub trait Conv<T: CommonBounds> {
    /// the output type of the conv operation
    type Output;
    /// Performs a 2D convolution operation on the input tensor.
    fn conv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(T::Vec) -> T::Vec>,
    ) -> Result<Self::Output, TensorError>;

    /// Performs a grouped 2D convolution operation on the input tensor.
    fn conv2d_group(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        groups: i64,
        activation: Option<fn(T::Vec) -> T::Vec>,
    ) -> Result<Self::Output, TensorError>;

    /// Performs a depthwise 2D convolution operation on the input tensor.
    fn dwconv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(T::Vec) -> T::Vec>,
    ) -> Result<Self::Output, TensorError>;

    /// Performs a 2D transposed convolution operation on the input tensor.
    fn conv2d_transpose(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        output_padding: [i64; 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError>;
}

/// trait for differentiable conv operations
pub trait ConvDiff<T: CommonBounds> {
    /// the output type of the conv operation
    type Output;

    /// Performs a 2D convolution operation on the input tensor.
    fn conv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError>;

    /// Performs a grouped 2D convolution operation on the input tensor.
    fn conv2d_group(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        groups: i64,
    ) -> Result<Self::Output, TensorError>;
}
