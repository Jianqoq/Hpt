# flipud
```rust
flipud(
    x: &Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Reverses the order of elements along axis 0 (rows) of the tensor. The tensor must be at least 1-dimensional.

## Parameters:
`x`: Input tensor with ndim >= 1

## Returns:
A new tensor with elements reversed along axis 0 (up/down flip).

## Examples:
```rust
use hpt::{ops::ShapeManipulate, Tensor, error::TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3])?;
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Flip up/down
    let b = a.flipud()?;
    // [[4, 5, 6],
    //  [1, 2, 3]]
    println!("{}", b);

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |