# tri
```rust
tri(
    n: usize,
    m: usize,
    k: i64,
    low_triangle: bool
) -> Result<Tensor<T>, TensorError>
```
Creates a tensor with ones at and below (or above) the k-th diagonal.

## Parameters:
`n`: Number of rows

`m`: Number of columns

`k`: The diagonal above or below which to fill with ones (0 represents the main diagonal)

`low_triangle`: If true, fill with ones below and on the k-th diagonal; if false, fill with ones above the k-th diagonal

## Returns:
A 2-D tensor of shape [n, m] with ones in the specified triangular region.

## Examples:
```rust
use hpt::{Tensor, TensorError, TensorCreator};
fn main() -> Result<(), TensorError> {
    // Lower triangular matrix (k=0)
    let a = Tensor::<f32>::tri(3, 3, 0, true)?;
    println!("{}", a);
    // [[1, 0, 0],
    //  [1, 1, 0],
    //  [1, 1, 1]]

    // Upper triangular matrix (k=0)
    let b = Tensor::<f32>::tri(3, 3, 0, false)?;
    println!("{}", b);
    // [[1, 1, 1],
    //  [0, 1, 1],
    //  [0, 0, 1]]

    // Lower triangular with offset (k=1)
    let c = Tensor::<f32>::tri(3, 4, 1, true)?;
    println!("{}", c);
    // [[1, 1, 0, 0],
    //  [1, 1, 1, 0],
    //  [1, 1, 1, 1]]

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |