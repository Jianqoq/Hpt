# conv2d_transpose
```rust
fn conv2d_transpose(
    x: &Tensor<T>,
    kernels: &Tensor<T>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    output_padding: [i64; 2],
    dilation: [i64; 2],
) -> Result<Tensor<T>, TensorError>
```
Performs a transpose 2D convolution operation with support for stride, padding, dilation, and activation functions.

## Parameters:
`x`: Input tensor with shape `[batch_size, height, width, in_channels]`

`kernels`: Transposed convolution kernels tensor with shape `[kernel_height, kernel_width, in_channels, out_channels]`

`steps`: Convolution stride as `[step_height, step_width]`

`padding`: Padding size as `[(padding_top, padding_bottom), (padding_left, padding_right)]`

`output_padding`: Padding size on the output

`dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::{
    common::TensorInfo,
    error::TensorError,
    ops::{Conv, Random},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // [batch_size, height, width, in_channels]
    let input = Tensor::<f32>::randn([1, 16, 16, 32])?;

    // [kernel_height, kernel_width, in_channels, out_channels]
    let kernels = Tensor::<f32>::randn([3, 3, 16, 32])?;

    // Perform transposed convolution to upsample the feature map
    let output = input.conv2d_transpose(
        &kernels,
        [2, 2],           // stride
        [(1, 1), (1, 1)], // padding
        [0, 0],           // output_padding
        [1, 1],           // dilation
    )?;

    println!("Output shape: {:?}", output.shape()); // [1, 31, 31, 16]
    Ok(())
}
```