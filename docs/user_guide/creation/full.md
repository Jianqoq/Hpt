# full
```rust
full(
    val: T,
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
Creates a new tensor of the specified shape, filled with a specified value.

## Parameters:
`val`: The value to fill the tensor with.
`shape`: The desired shape for the tensor.

## Returns:
A new tensor of the specified shape, filled with the given value.

## Examples:
```rust
use hpt_core::{Tensor, TensorError, TensorCreator};
fn main() -> Result<(), TensorError> {
    // Create a 2D tensor filled with 5.0
    let a = Tensor::<f32>::full(5.0, &[2, 3])?;
    println!("{}", a);
    // [[5, 5, 5],
    //  [5, 5, 5]]

    // Using a vector shape
    let shape = vec![2, 2];
    let b = Tensor::<i32>::full(42, shape)?;
    println!("{}", b);
    // [[42, 42],
    //  [42, 42]]

    Ok(())
}
```