use hpt_common::{error::base::TensorError, shape::shape::Shape};

/// trait for pooling that the output type is the same as the input type
pub trait NormalPooling {
    /// the output type is the same as the input type
    type Output;

    /// Performs a 2D max pooling operation on the input tensor, selecting the maximum value from each window.
    ///
    /// ## Parameters:
    /// `kernels`: Shape of the pooling window, typically `[kernel_height, kernel_width]`
    ///
    /// `steps`: Stride of the pooling operation as `[step_height, step_width]`
    ///
    /// `padding`: Padding size as `[(padding_top, padding_bottom), (padding_left, padding_right)]`
    ///
    /// `dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`
    ///
    /// ## Example:
    /// ```rust
    /// let input = Tensor::<f32>::randn([1, 32, 32, 16])?;
    /// let output = input.maxpool2d(
    ///     [2, 2],           // kernel size
    ///     [2, 2],           // stride
    ///     [(0, 0), (0, 0)], // padding
    ///     [1, 1],           // dilation
    /// )?; // shape: [1, 16, 16, 16]
    /// ```
    fn maxpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError>;

    /// Performs an adaptive max pooling operation on the input tensor, automatically determining the kernel size and stride to produce the specified output dimensions.
    ///
    /// ## Parameters:
    /// `output_size`: Desired output spatial dimensions as `[out_height, out_width]`
    ///
    /// ## Example:
    /// ```rust
    /// let input = Tensor::<f32>::randn([1, 32, 32, 16])?;
    /// let output = input.adaptive_maxpool2d([16, 16])?; // shape: [1, 16, 16, 16]
    /// let output2 = input.adaptive_maxpool2d([8, 8])?; // shape: [1, 8, 8, 16]
    /// ```
    fn adaptive_maxpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError>;
}

/// trait for pooling that the output type is the same as the input type
pub trait FloatOutPooling {
    /// the output type is the same as the input type
    type Output;

    /// Performs a 2D average pooling operation on the input tensor, computing the average value from each window.
    ///
    /// ## Parameters:
    /// `kernels`: Shape of the pooling window, typically `[kernel_height, kernel_width]`
    ///
    /// `steps`: Stride of the pooling operation as `[step_height, step_width]`
    ///
    /// `padding`: Padding size as `[(padding_top, padding_bottom), (padding_left, padding_right)]`
    ///
    /// `dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`
    ///
    /// ## Example:
    /// ```rust
    /// let input = Tensor::<f32>::randn([1, 32, 32, 16])?;
    /// let output = input.avgpool2d(
    ///     [2, 2],           // kernel size
    ///     [2, 2],           // stride
    ///     [(0, 0), (0, 0)], // padding
    ///     [1, 1],           // dilation
    /// )?; // shape: [1, 16, 16, 16]
    /// ```
    fn avgpool2d<S: Into<Shape>>(
        &self,
        kernels_shape: S,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError>;

    /// Performs an adaptive avg pooling operation on the input tensor, automatically determining the kernel size and stride to produce the specified output dimensions.
    ///
    /// ## Parameters:
    /// `output_size`: Desired output spatial dimensions as `[out_height, out_width]`
    ///
    /// ## Example:
    /// ```rust
    /// let input = Tensor::<f32>::randn([1, 32, 32, 16])?;
    /// let output = input.adaptive_avgpool2d([16, 16])?; // shape: [1, 16, 16, 16]
    /// let output2 = input.adaptive_avgpool2d([8, 8])?; // shape: [1, 8, 8, 16]
    /// ```
    fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<Self::Output, TensorError>;
}
