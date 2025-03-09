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
use hpt::{error::TensorError, ops::TensorCreator, Tensor};

fn main() -> Result<(), TensorError> {
    // Create a 3x4 matrix with ones on the main diagonal (k=0)
    let a = Tensor::<f32>::eye(3, 4, 0)?;
    println!("Main diagonal:\n{}", a);
    // Output:
    // [[1, 0, 0, 0],
    //  [0, 1, 0, 0],
    //  [0, 0, 1, 0]]

    // Create a 3x4 matrix with ones on the first superdiagonal (k=1)
    let b = Tensor::<f32>::eye(3, 4, 1)?;
    println!("First superdiagonal:\n{}", b);
    // Output:
    // [[0, 1, 0, 0],
    //  [0, 0, 1, 0],
    //  [0, 0, 0, 1]]

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |