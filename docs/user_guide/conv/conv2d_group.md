# conv2d_group
```rust
fn conv2d_group(
        x: &Tensor<T>,
        kernels: &Tensor<T>,
        bias: Option<&Tensor<T>>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        groups: i64,
        activation: Option<fn(T::Vec) -> T::Vec>,
    ) -> Result<Tensor<T>, TensorError>
```
Performs a grouped 2D convolution operation, which divides input channels into groups and performs separate convolutions on each group.

## Parameters:
`x`: Input tensor with shape [batch_size, height, width, in_channels]

`kernels`: Convolution kernels tensor with shape [kernel_height, kernel_width, in_channels/groups, out_channels]

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
    let input = Tensor::<f32>::randn([1, 32, 32, 32])?;

    // [kernel_height, kernel_width, in_channels/groups, out_channels]
    // For 4 groups with 32 input channels and 16 output channels
    let kernels = Tensor::<f32>::randn([3, 3, 8, 16])?;

    // Create bias
    let bias = Tensor::<f32>::randn([16])?;

    // Perform grouped convolution with 4 groups
    let output = input.conv2d_group(
        &kernels,
        Some(&bias),
        [1, 1],           // stride
        [(1, 1), (1, 1)], // padding
        [1, 1],           // dilation
        4,                // groups
        None,             // no activation function
    )?;

    println!("Output shape: {:?}", output.shape()); // [1, 32, 32, 16]
    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |