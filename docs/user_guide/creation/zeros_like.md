# zeros_like
```rust
zeros_like(
    x: &Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Creates a new tensor filled with zeros with the same shape as the input tensor.

## Parameters:
`x`: The input tensor whose shape will be used.

## Returns:
A new tensor of zeros with the same shape as the input tensor.

## Examples:
```rust
use hpt::{ShapeManipulate, Tensor, TensorCreator, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    println!("a: {}", a);
    // [[1, 2],
    //  [3, 4]]

    // Create a tensor of zeros with same shape
    let b = a.zeros_like()?;
    println!("b: {}", b);
    // [[0, 0],
    //  [0, 0]]

    Ok(())
}
```