# full_like
```rust
full_like(
    x: &Tensor<T>,
    val: T
) -> Result<Tensor<T>, TensorError>
```
Creates a new tensor filled with a specified value with the same shape as the input tensor.

## Parameters:
`x`: The input tensor whose shape will be used.
`val`: The value to fill the tensor with.

## Returns:
A new tensor filled with the specified value with the same shape as the input tensor.

## Examples:
```rust
use hpt::{ShapeManipulate, Tensor, TensorCreator, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    println!("a: {}", a);
    // [[1, 2],
    //  [3, 4]]

    // Create a tensor filled with 7.0 with same shape
    let b = a.full_like(7.0)?;
    println!("b: {}", b);
    // [[7, 7],
    //  [7, 7]]

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |