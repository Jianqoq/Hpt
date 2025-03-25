# batchnorm_conv2d
```rust
fn batchnorm_conv2d(
        x: &Tensor<T>,
        kernels: &Tensor<T>,
        mean: &Tensor<T>,
        var: &Tensor<T>,
        gamma: &Tensor<T>,
        beta: &Tensor<T>,
        bias: Option<&Tensor<T>>,
        eps: T,
        steps: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        algo: Option<ConvAlgo>,
    ) -> Result<Tensor<T>, TensorError>
```
Performs a 2D convolution operation followed by batch normalization in a single fused operation for improved performance.

## Parameters:
`x`: Input tensor with shape `[batch_size, height, width, in_channels]`

`kernels`: Convolution kernels tensor with shape `[out_channels, kernel_height, kernel_width, in_channels]`

`mean`: Mean values for batch normalization with shape `[out_channels]`

`var`: Variance values for batch normalization with shape `[out_channels]`

`gamma`: Scale parameter for batch normalization with shape `[out_channels]`

`beta`: Shift parameter for batch normalization with shape `[out_channels]`

`bias`: Optional bias tensor with shape `[out_channels]`

`eps`: Small constant added to the variance for numerical stability

`steps`: Convolution stride as `[step_height, step_width]`

`padding`: Padding size as `[padding_height, padding_width]`

`dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`

`algo`: Optional algorithm to use, None will auto select

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::{
    backend::Cuda,
    common::TensorInfo,
    error::TensorError,
    ops::{CudaConvBatchNorm, Random, TensorCreator},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // [batch_size, height, width, in_channels]
    let input = Tensor::<f32, Cuda>::randn([1, 32, 32, 3])?;

    // [out_channels, kernel_height, kernel_width, in_channels]
    let kernels = Tensor::<f32, Cuda>::randn([16, 3, 3, 3])?;

    // Batch normalization parameters
    let mean = Tensor::<f32, Cuda>::zeros([16])?;
    let var = Tensor::<f32, Cuda>::ones([16])?;
    let gamma = Tensor::<f32, Cuda>::ones([16])?;
    let beta = Tensor::<f32, Cuda>::zeros([16])?;

    // Optional convolution bias
    let bias = Tensor::<f32, Cuda>::zeros([16])?;

    // Perform fused convolution with batch normalization
    let output = input.batchnorm_conv2d(
        &kernels,
        &mean,
        &var,
        &gamma,
        &beta,
        Some(&bias),
        1e-5,   // epsilon
        [1, 1], // stride
        [1, 1], // padding
        [1, 1], // dilation
        None,   // auto select algo
    )?;

    println!("Output shape: {:?}", output.shape()); // [1, 32, 32, 16]
    Ok(())
}
```