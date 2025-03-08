# ones_like
```rust
ones_like(
    x: &Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Creates a new tensor filled with ones with the same shape as the input tensor.

## Parameters:
`x`: The input tensor whose shape will be used.

## Returns:
A new tensor of ones with the same shape as the input tensor.

## Examples:
```rust
use hpt::{
    error::TensorError,
    ops::{ShapeManipulate, TensorCreator},
    Tensor,
};
fn main() -> Result<(), TensorError> {
    // Create a tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    println!("a: {}", a);
    // [[1, 2],
    //  [3, 4]]

    // Create a tensor of ones with same shape
    let b = a.ones_like()?;
    println!("b: {}", b);
    // [[1, 1],
    //  [1, 1]]

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |