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
use hpt_core::{Tensor, TensorError, TensorCreator};
fn main() -> Result<(), TensorError> {
    // Create a 3x3 identity matrix (k=0)
    let a = Tensor::<f32>::eye(3, 3, 0)?;
    println!("{}", a);
    // [[1, 0, 0],
    //  [0, 1, 0],
    //  [0, 0, 1]]

    // Create a 3x4 matrix with ones on first superdiagonal (k=1)
    let b = Tensor::<f32>::eye(3, 4, 1)?;
    println!("{}", b);
    // [[0, 1, 0, 0],
    //  [0, 0, 1, 0],
    //  [0, 0, 0, 1]]

    // Create a 4x3 matrix with ones on main diagonal
    let c = Tensor::<f32>::eye(4, 3, 0)?;
    println!("{}", c);
    // [[1, 0, 0],
    //  [0, 1, 0],
    //  [0, 0, 1],
    //  [0, 0, 0]]

    Ok(())
}
```