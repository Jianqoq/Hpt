# avgpool2d
```rust
fn avgpool2d(
    x: &Tensor<T>,
    kernels_shape: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
) -> Result<Tensor<C>, TensorError>
```
Performs a 2D average pooling operation on the input tensor, computing the average value from each window.

## Parameters:
`x`: Input tensor with shape [batch_size, height, width, in_channels]

`kernels`: Shape of the pooling window, typically [kernel_height, kernel_width]

`steps`: Stride of the pooling operation as [step_height, step_width]

`padding`: Padding size as [(padding_top, padding_bottom), (padding_left, padding_right)]

`dilation`: Kernel dilation factors as [dilation_height, dilation_width]

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{
    common::TensorInfo,
    error::TensorError,
    ops::{FloatOutPooling, Random},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // [batch_size, height, width, channels]
    let input = Tensor::<f32>::randn([1, 32, 32, 16])?;

    // Perform avg pooling with 2x2 kernel and stride 2
    let output = input.avgpool2d(
        [2, 2],           // kernel size
        [2, 2],           // stride
        [(0, 0), (0, 0)], // padding
        [1, 1],           // dilation
    )?;

    println!("Output shape: {:?}", output.shape()); // [1, 16, 16, 16]
    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |