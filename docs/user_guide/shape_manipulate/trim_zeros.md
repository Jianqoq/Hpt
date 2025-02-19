# trim_zeros
```rust
trim_zeros(
    x: &Tensor<T>,
    trim: &str
) -> Result<Tensor<T>, TensorError>
```
Removes zeros from the beginning and/or end of a 1-D tensor.

## Parameters:
`x`: Input tensor (must be 1-dimensional)

`trim`: A string specifying which zeros to remove:
- 'f': remove leading zeros (from front)
- 'b': remove trailing zeros (from back)
- 'fb' or 'bf': remove both leading and trailing zeros

## Returns:
A new tensor with zeros trimmed from the specified ends.

## Examples:
```rust
use hpt::{ShapeManipulate, Tensor, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 1D tensor with zeros
    let a = Tensor::<f32>::new(&[0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    
    // Trim zeros from front
    let b = a.trim_zeros("f")?;
    // [1, 2, 3, 0, 0]
    println!("{}", b);

    // Trim zeros from back
    let c = a.trim_zeros("b")?;
    // [0, 0, 1, 2, 3]
    println!("{}", c);

    // Trim zeros from both ends
    let d = a.trim_zeros("fb")?;
    // [1, 2, 3]
    println!("{}", d);

    // Will raise an error for 2D tensor
    let e = Tensor::<f32>::new(&[0.0, 1.0, 0.0, 1.0]).reshape(&[2, 2])?;
    assert!(e.trim_zeros("f").is_err());

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |