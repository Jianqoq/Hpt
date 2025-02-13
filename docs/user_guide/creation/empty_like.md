# empty_like
```rust
empty_like(
    x: &Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Creates a new uninitialized tensor with the same shape as the input tensor.

## Parameters:
`x`: The input tensor whose shape will be used.

## Returns:
A new uninitialized tensor with the same shape as the input tensor.

## Examples:
```rust
use hpt::{ShapeManipulate, Tensor, TensorCreator, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create a tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]).reshape(&[3, 1])?;
    println!("Shape a: {:?}", a.shape()); // prints: Shape a: [3, 1]

    // Create an empty tensor with same shape
    let b = a.empty_like()?;
    println!("Shape b: {:?}", b.shape()); // prints: Shape b: [3, 1]

    Ok(())
}
```