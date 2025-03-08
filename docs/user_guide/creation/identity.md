# identity
```rust
identity(
    n: usize
) -> Result<Tensor<T>, TensorError>
```
Creates a 2-D identity tensor (1's on the main diagonal and 0's elsewhere).

## Parameters:
`n`: Number of rows and columns

## Returns:
A square 2-D tensor of shape [n, n] with ones on the main diagonal.

## Examples:
```rust
use hpt::{error::TensorError, ops::TensorCreator, Tensor};
fn main() -> Result<(), TensorError> {
    // Create a 3x3 identity matrix
    let a = Tensor::<f32>::identity(3)?;
    println!("{}", a);
    // [[1, 0, 0],
    //  [0, 1, 0],
    //  [0, 0, 1]]

    // Create a 2x2 identity matrix
    let b = Tensor::<f32>::identity(2)?;
    println!("{}", b);
    // [[1, 0],
    //  [0, 1]]

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |