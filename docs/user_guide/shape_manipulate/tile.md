# tile
```rust
tile(
    x: &Tensor<T>,
    repeats: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | i64
) -> Result<Tensor<T>, TensorError>
```
Constructs a new tensor by repeating the input tensor along specified dimensions.

## Parameters:
`x`: Input tensor

`repeats`: The number of repetitions for each dimension. If `repeats` has fewer dimensions than the input tensor, it is padded with 1s. If `repeats` has more dimensions than the input tensor, the input tensor is padded with dimensions of size 1.

## Returns:
A new tensor containing the input tensor repeated according to `repeats`.

## Examples:
```rust
use hpt::{ops::ShapeManipulate, Tensor, error::TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    // [[1, 2],
    //  [3, 4]]

    // Tile with repeats [2, 1] (repeat rows twice)
    let b = a.tile(&[2, 1])?;
    // [[1, 2],
    //  [3, 4],
    //  [1, 2],
    //  [3, 4]]
    println!("{}", b);

    // Tile with repeats [1, 2] (repeat columns twice)
    let c = a.tile(&[1, 2])?;
    // [[1, 2, 1, 2],
    //  [3, 4, 3, 4]]
    println!("{}", c);

    // Tile with repeats [2, 2] (repeat both dimensions twice)
    let d = a.tile(&[2, 2])?;
    // [[1, 2, 1, 2],
    //  [3, 4, 3, 4],
    //  [1, 2, 1, 2],
    //  [3, 4, 3, 4]]
    println!("{}", d);

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |