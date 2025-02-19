# hsplit
```rust
hsplit(
    x: &Tensor<T>,
    indices: &[i64]
) -> Result<Vec<Tensor<T>>, TensorError>
```
Splits a tensor into multiple sub-tensors horizontally (along axis 1). The tensor must be at least 2-dimensional.

## Parameters:
`x`: Input tensor with ndim >= 2

`indices`: The indices where the splits should occur along axis 1 (columns)

## Returns:
A vector of sub-tensors created by splitting the input tensor horizontally.

## Examples:
```rust
use hpt::{ShapeManipulate, Tensor, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor with shape [2, 4]
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0]).reshape(&[2, 4])?;
    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8]]

    // Split horizontally at index [2]
    let splits = a.hsplit(&[2])?;
    // splits[0]:
    // [[1, 2],
    //  [5, 6]]
    // splits[1]:
    // [[3, 4],
    //  [7, 8]]
    for (i, split) in splits.iter().enumerate() {
        println!("Split {}: {}", i, split);
    }

    // Will raise an error for 1D tensor
    let b = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]);
    assert!(b.hsplit(&[2]).is_err());

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |