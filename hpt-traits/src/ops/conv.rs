use hpt_common::error::base::TensorError;

use crate::tensor::CommonBounds;

/// trait for conv operations
pub trait Conv<T: CommonBounds> {
    /// the output type of the conv operation
    type Output;
    /// Performs a 2D convolution operation with support for stride, padding, dilation, and activation functions.
    ///
    /// ## Parameters:
    /// `self`: Input tensor with shape `[batch_size, height, width, in_channels]`
    ///
    /// `kernels`: Convolution kernels tensor with shape `[kernel_height, kernel_width, in_channels, out_channels]`
    ///
    /// `bias`: Optional bias tensor with shape `[out_channels]`
    ///
    /// `steps`: Convolution stride as `[step_height, step_width]`
    ///
    /// `padding`: Padding size as `[(padding_top, padding_bottom), (padding_left, padding_right)]`
    ///
    /// `dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`
    ///
    /// `activation`: Optional activation function applied to the convolution result
    ///
    /// ## Example:
    /// ```rust
    /// let input = Tensor::<f32>::randn([1, 32, 32, 3])?;
    /// let kernels = Tensor::<f32>::randn([3, 3, 3, 16])?;
    /// let bias = Tensor::<f32>::randn([16])?;
    /// let output = input.conv2d(
    ///     &kernels,
    ///     Some(&bias),
    ///     [2, 2],           // stride
    ///     [(0, 0), (0, 0)], // padding
    ///     [1, 1],           // dilation
    ///     None,             // no activation function
    /// )?; // [batch_size, out_height, out_width, out_channels]
    /// ```
    fn conv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(T::Vec) -> T::Vec>,
    ) -> Result<Self::Output, TensorError>;

    /// Performs a grouped 2D convolution operation, which divides input channels into groups and performs separate convolutions on each group.
    ///
    /// ## Parameters:
    /// `self`: Input tensor with shape `[batch_size, height, width, in_channels]`
    ///
    /// `kernels`: Convolution kernels tensor with shape `[kernel_height, kernel_width, in_channels/groups, out_channels]`
    ///
    /// `bias`: Optional bias tensor with shape `[out_channels]`
    ///
    /// `steps`: Convolution stride as `[step_height, step_width]`
    ///
    /// `padding`: Padding size as `[(padding_top, padding_bottom), (padding_left, padding_right)]`
    ///
    /// `dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`
    ///
    /// `groups`: Number of groups to use
    ///
    /// `activation`: Optional activation function applied to the convolution result
    ///
    /// ## Example:
    /// ```rust
    /// let input = Tensor::<f32>::randn([1, 32, 32, 32])?;
    /// let kernels = Tensor::<f32>::randn([3, 3, 8, 16])?;
    /// let bias = Tensor::<f32>::randn([16])?;
    /// let output = input.conv2d_group(
    ///     &kernels,
    ///     Some(&bias),
    ///     [1, 1],           // stride
    ///     [(1, 1), (1, 1)], // padding
    ///     [1, 1],           // dilation
    ///     groups,           // number of groups
    ///     None,             // no activation function
    /// )?; // [batch_size, out_height, out_width, out_channels]
    /// ```
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

    /// Performs a depthwise 2D convolution operation with support for stride, padding, dilation, and activation functions.
    ///
    /// ## Parameters:
    /// `self`: Input tensor with shape `[batch_size, height, width, in_channels]`
    ///
    /// `kernels`: Convolution kernels tensor with shape `[kernel_height, kernel_width, in_channels, out_channels]`
    ///
    /// `bias`: Optional bias tensor with shape `[out_channels]`
    ///
    /// `steps`: Convolution stride as `[step_height, step_width]`
    ///
    /// `padding`: Padding size as `[(padding_top, padding_bottom), (padding_left, padding_right)]`
    ///
    /// `dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`
    ///
    /// `activation`: Optional activation function applied to the convolution result
    ///
    /// ## Example:
    /// ```rust
    /// let input = Tensor::<f32>::randn([1, 32, 32, 16])?;
    /// let kernels = Tensor::<f32>::randn([3, 3, 1, 16])?;
    /// let bias = Tensor::<f32>::randn([16])?;
    /// let output = input.dwconv2d(
    ///     &kernels,
    ///     Some(&bias),
    ///     [2, 2],           // stride
    ///     [(0, 0), (0, 0)], // padding
    ///     [1, 1],           // dilation
    ///     None,             // no activation function
    /// )?; // shape([1, 15, 15, 16])
    /// ```
    fn dwconv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(T::Vec) -> T::Vec>,
    ) -> Result<Self::Output, TensorError>;

    /// Performs a transpose 2D convolution operation with support for stride, padding, dilation, and activation functions.
    ///
    /// ## Parameters:
    /// `self`: Input tensor with shape `[batch_size, height, width, in_channels]`
    ///
    /// `kernels`: Transposed convolution kernels tensor with shape `[kernel_height, kernel_width, in_channels, out_channels]`
    ///
    /// `steps`: Convolution stride as `[step_height, step_width]`
    ///
    /// `padding`: Padding size as `[(padding_top, padding_bottom), (padding_left, padding_right)]`
    ///
    /// `output_padding`: Padding size on the output
    ///
    /// `dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`
    ///
    /// ## Example:
    /// ```rust
    /// // [batch_size, height, width, in_channels]
    /// let input = Tensor::<f32>::randn([1, 16, 16, 32])?;
    /// // [kernel_height, kernel_width, in_channels, out_channels]
    /// let kernels = Tensor::<f32>::randn([3, 3, 16, 32])?;
    /// let output = input.conv2d_transpose(
    ///     &kernels,
    ///     [2, 2],           // stride
    ///     [(1, 1), (1, 1)], // padding
    ///     [0, 0],           // output_padding
    ///     [1, 1],           // dilation
    /// )?; // shape([1, 31, 31, 16])
    /// ```
    fn conv2d_transpose(
        &self,
        kernels: &Self::Output,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        output_padding: [i64; 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, TensorError>;
}

/// trait for conv operations with batch normalization
pub trait ConvBatchNorm<T: CommonBounds> {
    /// the output type of the conv operation
    type Output;

    /// Performs a 2D convolution operation followed by batch normalization in a single fused operation for improved performance.
    ///
    /// ## Parameters:
    /// `self`: Input tensor with shape `[batch_size, height, width, in_channels]`
    ///
    /// `kernels`: Convolution kernels tensor with shape `[kernel_height, kernel_width, in_channels, out_channels]`
    ///
    /// `mean`: Mean values for batch normalization with shape [out_channels]
    ///
    /// `var`: Variance values for batch normalization with shape [out_channels]
    ///
    /// `gamma`: Scale parameter for batch normalization with shape [out_channels]
    ///
    /// `beta`: Shift parameter for batch normalization with shape [out_channels]
    ///
    /// `bias`: Optional bias tensor with shape `[out_channels]`
    ///
    /// `eps`: Small constant added to the variance for numerical stability
    ///
    /// `steps`: Convolution stride as `[step_height, step_width]`
    ///
    /// `padding`: Padding size as `[(padding_top, padding_bottom), (padding_left, padding_right)]`
    ///
    /// `dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`
    ///
    /// `activation`: Optional activation function applied to the convolution result
    ///
    /// ## Example:
    /// ```rust
    /// let input = Tensor::<f32>::randn([1, 32, 32, 3])?;
    /// let kernels = Tensor::<f32>::randn([3, 3, 3, 16])?;
    /// let mean = Tensor::<f32>::zeros([16])?;
    /// let var = Tensor::<f32>::ones([16])?;
    /// let gamma = Tensor::<f32>::ones([16])?;
    /// let beta = Tensor::<f32>::zeros([16])?;
    /// let bias = Tensor::<f32>::randn([16])?;
    /// let output = input.batchnorm_conv2d(
    ///     &kernels,
    ///     &mean,
    ///     &var,
    ///     &gamma,
    ///     &beta,
    ///     Some(&bias),
    ///     1e-5,             // epsilon
    ///     [1, 1],           // stride
    ///     [(1, 1), (1, 1)], // padding
    ///     [1, 1],           // dilation
    ///     None,             // no activation function
    /// )?; // shape([1, 32, 32, 16])
    /// ```
    fn batchnorm_conv2d(
        &self,
        kernels: &Self::Output,
        mean: &Self::Output,
        var: &Self::Output,
        gamma: &Self::Output,
        beta: &Self::Output,
        bias: Option<&Self::Output>,
        eps: T,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(T::Vec) -> T::Vec>,
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
