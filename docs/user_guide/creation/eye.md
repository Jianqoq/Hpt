# eye
```rust
eye(
    n: usize,
    m: usize,
    k: usize
) -> Result<Tensor<T>, TensorError>
```
Creates a 2-D tensor with ones on the k-th diagonal and zeros elsewhere.

## Parameters:
`n`: Number of rows
`m`: Number of columns
`k`: Index of the diagonal (0 represents the main diagonal, positive values are above the main diagonal)

## Returns:
A 2-D tensor of shape [n, m] with ones on the k-th diagonal.

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