# ones
```rust
ones(
    shape: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | &Shape
        | Shape
) -> Result<Tensor<T>, TensorError>
```
Creates a new tensor of the specified shape, filled with ones.

## Parameters:
`shape`: The desired shape for the tensor.

## Returns:
A new tensor of the specified shape, filled with ones.

## Examples:
```rust
use tensor_dyn::{Tensor, TensorCreator, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor of ones
    let a = Tensor::<f32>::ones(&[2, 3])?;
    println!("{}", a);
    // [[1, 1, 1],
    //  [1, 1, 1]]

    // Using a vector
    let shape = vec![2, 2];
    let b = Tensor::<i32>::ones(shape)?;
    println!("{}", b);
    // [[1, 1],
    //  [1, 1]]

    Ok(())
}
```