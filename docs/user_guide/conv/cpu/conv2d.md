# conv2d
```rust
fn conv2d(
    x: &Tensor<T>,
    kernels: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
) -> Result<Tensor<T>, TensorError>
```
Performs a 2D convolution operation with support for stride, padding, dilation, and activation functions.

## Parameters:
`x`: Input tensor with shape `[batch_size, height, width, in_channels]`

`kernels`: Convolution kernels tensor with shape `[kernel_height, kernel_width, in_channels, out_channels]`

`bias`: Optional bias tensor with shape `[out_channels]`

`steps`: Convolution stride as `[step_height, step_width]`

`padding`: Padding size as `[(padding_top, padding_bottom), (padding_left, padding_right)]`

`dilation`: Kernel dilation factors as `[dilation_height, dilation_width]`

`post_scalar`: Optional post function applied to each of the scalar result

`post_vec`: Optional post_vec function applied to each of the vector result

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
    let input = Tensor::<f32>::randn([1, 32, 32, 3])?;

    // [kernel_height, kernel_width, in_channels, out_channels]
    let kernels = Tensor::<f32>::randn([3, 3, 3, 16])?;

    // Create bias
    let bias = Tensor::<f32>::randn([16])?;

    // Perform convolution with stride 2, no padding, no dilation
    let output = input.conv2d(
        &kernels,
        Some(&bias),
        [2, 2],           // stride
        [(0, 0), (0, 0)], // padding
        [1, 1],           // dilation
        None,             // no activation function
        None,             // no activation function
    )?;

    println!("Output shape: {:?}", output.shape()); // [batch_size, out_height, out_width, out_channels]
    Ok(())
}
```