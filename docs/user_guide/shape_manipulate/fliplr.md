# fliplr
```rust
fliplr(
    x: &Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Reverses the order of elements along axis 1 (columns) of the tensor. The tensor must be at least 2-dimensional.

## Parameters:
`x`: Input tensor with ndim >= 2

## Returns:
A new tensor with elements reversed along axis 1 (left/right flip).

## Examples:
```rust
use hpt::{ops::ShapeManipulate, Tensor, error::TensorError, common::TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3])?;
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Flip left/right
    let b = a.fliplr()?;
    // [[3, 2, 1],
    //  [6, 5, 4]]
    println!("{}", b);

    // Will raise an error for 1D tensor
    let c = Tensor::<f32>::new(&[1.0, 2.0, 3.0]).reshape(&[3])?;
    assert!(c.fliplr().is_err());

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |