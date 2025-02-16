# flip
```rust
flip(
    x: &Tensor<T>,
    axes: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | i64
) -> Result<Tensor<T>, TensorError>
```
Reverses the order of elements in the tensor along the specified axes.

## Parameters:
`x`: Input tensor

`axes`: The axes along which to flip the tensor. Can be a single axis or multiple axes.

## Returns:
A new tensor with elements reversed along the specified axes.

## Examples:
```rust
use hpt::{ShapeManipulate, Tensor, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3])?;
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Flip along axis 0 (rows)
    let b = a.flip(0)?;
    // [[4, 5, 6],
    //  [1, 2, 3]]
    println!("{}", b);

    // Flip along axis 1 (columns)
    let c = a.flip(1)?;
    // [[3, 2, 1],
    //  [6, 5, 4]]
    println!("{}", c);

    // Flip along both axes
    let d = a.flip(&[0, 1])?;
    // [[6, 5, 4],
    //  [3, 2, 1]]
    println!("{}", d);

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |