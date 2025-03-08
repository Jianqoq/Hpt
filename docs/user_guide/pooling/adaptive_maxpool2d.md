# adaptive_maxpool2d
```rust
fn adaptive_maxpool2d(
    x: &Tensor<T>,
    output_size: [i64; 2]
) -> Result<Tensor<T>, TensorError>
```
Performs an adaptive max pooling operation on the input tensor, automatically determining the kernel size and stride to produce the specified output dimensions.

## Parameters:
`x`: Input tensor with shape [batch_size, height, width, channels]

`output_size`: Desired output spatial dimensions as [out_height, out_width]

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::{
    common::TensorInfo,
    error::TensorError,
    ops::{NormalPooling, Random},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // [batch_size, height, width, channels]
    let input = Tensor::<f32>::randn([1, 32, 32, 16])?;

    // Perform adaptive max pooling to get 16x16 output
    let output = input.adaptive_maxpool2d([16, 16])?;

    println!("Output shape: {:?}", output.shape()); // [1, 16, 16, 16]

    // Resize to a different output size
    let output2 = input.adaptive_maxpool2d([8, 8])?;

    println!("Output2 shape: {:?}", output2.shape()); // [1, 8, 8, 16]
    Ok(())
}
```

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |