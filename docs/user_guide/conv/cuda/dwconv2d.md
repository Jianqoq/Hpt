# dwconv2d
```rust
fn dwconv2d(
    x: &Tensor<T>,
    kernels: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    steps: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    algo: Option<ConvAlgo>
) -> Result<Tensor<T>, TensorError>
```
Performs a depthwise 2D convolution operation with support for stride, padding, dilation.

## Parameters:
`x`: Input tensor with shape `[batch_size, height, width, in_channels]`

`kernels`: Convolution kernels tensor with shape `[out_channel, kernel_height, kernel_width, in_channels]`

`bias`: Optional bias tensor with shape `[out_channels]`

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
    ops::{CudaConv, Random},
    Tensor,
};

fn main() -> anyhow::Result<(), TensorError> {
    // [batch_size, height, width, in_channels]
    let input = Tensor::<f32, Cuda>::randn([1, 32, 32, 16])?;

    // [out_channel, kernel_height, kernel_width, in_channels]
    let kernels = Tensor::<f32, Cuda>::randn([16, 3, 3, 1])?;

    // Create bias
    let bias = Tensor::<f32, Cuda>::randn([16])?;

    let output = input.dwconv2d(
        &kernels,
        Some(&bias),
        [2, 2], // stride
        [0, 0], // padding
        [1, 1], // dilation
        None,   // auto select algo
    )?;

    println!("Output shape: {:?}", output.shape());
    Ok(())
}
```