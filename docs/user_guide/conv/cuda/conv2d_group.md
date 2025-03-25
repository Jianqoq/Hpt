# conv2d_group
```rust
fn conv2d_group(
        x: &Tensor<T>,
        kernels: &Tensor<T>,
        bias: Option<&Tensor<T>>,
        steps: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        groups: i64,
        algo: Option<ConvAlgo>
    ) -> Result<Tensor<T>, TensorError>
```
Performs a grouped 2D convolution operation, which divides input channels into groups and performs separate convolutions on each group.

## Parameters:
`x`: Input tensor with shape `[batch_size, height, width, in_channels]`

`kernels`: Convolution kernels tensor with shape `[out_channels, kernel_height, kernel_width, in_channels/groups]`

`bias`: Optional bias tensor with shape `[out_channels]`

`steps`: Convolution stride as `[step_height, step_width]`

`padding`: Padding size as `[padding_height, padding_width]`

`dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`

`groups`: Number of groups to use

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

fn main() -> Result<(), TensorError> {
    // [batch_size, height, width, in_channels]
    let input = Tensor::<f32, Cuda>::randn([1, 32, 32, 32])?;

    // [out_channels, kernel_height, kernel_width, in_channels/groups]
    // For 4 groups with 32 input channels and 16 output channels
    let kernels = Tensor::<f32, Cuda>::randn([16, 3, 3, 8])?;

    // Create bias
    let bias = Tensor::<f32, Cuda>::randn([16])?;

    // Perform grouped convolution with 4 groups
    let output = input.conv2d_group(
        &kernels,
        Some(&bias),
        [1, 1], // stride
        [1, 1], // padding
        [1, 1], // dilation
        4,      // groups
        None,   // auto select algo
    )?;

    println!("Output shape: {:?}", output.shape()); // [1, 32, 32, 16]
    Ok(())
}
```