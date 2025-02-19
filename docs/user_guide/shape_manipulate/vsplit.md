# vsplit
```rust
vsplit(
    x: &Tensor<T>,
    indices: &[i64]
) -> Result<Vec<Tensor<T>>, TensorError>
```
Splits a tensor into multiple sub-tensors vertically (along axis 0). The tensor must be at least 1-dimensional.

## Parameters:
`x`: Input tensor with ndim >= 1

`indices`: The indices where the splits should occur along axis 0 (rows)

## Returns:
A vector of sub-tensors created by splitting the input tensor vertically.

## Examples:
```rust
use hpt::{ShapeManipulate, Tensor, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor with shape [4, 2]
    let a = Tensor::<f32>::new(&[1.0, 2.0,
                                3.0, 4.0,
                                5.0, 6.0,
                                7.0, 8.0]).reshape(&[4, 2])?;
    // [[1, 2],
    //  [3, 4],
    //  [5, 6],
    //  [7, 8]]

    // Split vertically at index [2]
    let splits = a.vsplit(&[2])?;
    // splits[0]:
    // [[1, 2],
    //  [3, 4]]
    // splits[1]:
    // [[5, 6],
    //  [7, 8]]
    for (i, split) in splits.iter().enumerate() {
        println!("Split {}: {}", i, split);
    }

    // Works with 1D tensor too
    let b = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]);
    let splits = b.vsplit(&[2])?;
    // splits[0]: [1, 2]
    // splits[1]: [3, 4]
    for (i, split) in splits.iter().enumerate() {
        println!("Split {}: {}", i, split);
    }

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |