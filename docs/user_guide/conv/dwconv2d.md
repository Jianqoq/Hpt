# dwconv2d
```rust
fn dwconv2d(
    x: &Tensor<T>,
    kernels: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    activation: Option<fn(T::Vec) -> T::Vec>,
) -> Result<Tensor<T>, TensorError>
```
Performs a depthwise 2D convolution operation with support for stride, padding, dilation, and activation functions.

## Parameters:
`x`: Input tensor with shape [batch_size, height, width, in_channels]

`kernels`: Convolution kernels tensor with shape [kernel_height, kernel_width, in_channels, out_channels]

`bias`: Optional bias tensor with shape [out_channels]

`steps`: Convolution stride as [step_height, step_width]

`padding`: Padding size as [(padding_top, padding_bottom), (padding_left, padding_right)]

`dilation`: Kernel dilation factors as [dilation_height, dilation_width]

`activation`: Optional activation function applied to the convolution result

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::{Conv, Random, Tensor, TensorError, TensorInfo};

fn main() -> Result<(), TensorError> {
    // [batch_size, height, width, in_channels]
    let input = Tensor::<f32>::randn([1, 32, 32, 16])?;

    // [kernel_height, kernel_width, in_channels, out_channels]
    let kernels = Tensor::<f32>::randn([3, 3, 1, 16])?;

    // Create bias
    let bias = Tensor::<f32>::randn([16])?;

    let output = input.dwconv2d(
        &kernels,
        Some(&bias),
        [2, 2],           // stride
        [(0, 0), (0, 0)], // padding
        [1, 1],           // dilation
        None,             // no activation function
    )?;

    println!("Output shape: {:?}", output.shape());
    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |