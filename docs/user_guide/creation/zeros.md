# zeros
```rust
zeros(
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
Creates a new tensor of the specified shape, filled with zeros.

## Parameters:
`shape`: The desired shape for the tensor.

## Returns:
A new tensor of the specified shape, filled with zeros.

## Examples:
```rust
use hpt_core::{Tensor, TensorCreator, TensorError};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor of zeros
    let a = Tensor::<f32>::zeros(&[2, 3])?;
    println!("{}", a);
    // [[0, 0, 0],
    //  [0, 0, 0]]

    // Using a vector
    let shape = vec![2, 2];
    let b = Tensor::<i32>::zeros(shape)?;
    println!("{}", b);
    // [[0, 0],
    //  [0, 0]]

    Ok(())
}
```