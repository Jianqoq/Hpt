use hpt_common::{error::base::TensorError, shape::shape::Shape};

/// trait for pooling that the output type is the same as the input type
pub trait NormalPooling {
    /// the output type is the same as the input type
    type Output;

    /// Performs a 2D max pooling operation on the input tensor.
    fn maxpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError>;

    /// Performs a adaptive max pooling operation on the input tensor.
    fn adaptive_maxpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError>;
}

/// trait for pooling that the output type is the same as the input type
pub trait FloatOutPooling {
    /// the output type is the same as the input type
    type Output;

    /// Performs a 2D average pooling operation on the input tensor.
    fn avgpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError>;

    /// Performs a adaptive average pooling operation on the input tensor.
    fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError>;
}
