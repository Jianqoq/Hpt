# empty
```rust
empty(
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
Creates a new uninitialized tensor with the specified shape. The tensor's values will be whatever was in memory at the time of allocation.

## Parameters:
`shape`: The desired shape for the tensor.

## Returns:
A new uninitialized tensor with the specified shape.

## Examples:
```rust
use tensor_dyn::{Tensor, TensorCreator, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create an empty 2D tensor
    let a = Tensor::<f32>::empty(&[2, 3])?;
    println!("Shape: {:?}", a.shape()); // prints: Shape: [2, 3]

    // Using a vector
    let shape = vec![3, 4, 5];
    let b = Tensor::<f32>::empty(shape)?;
    println!("Shape: {:?}", b.shape()); // prints: Shape: [3, 4, 5]

    // Using an array
    let c = Tensor::<f32>::empty([2, 2])?;
    println!("Shape: {:?}", c.shape()); // prints: Shape: [2, 2]

    // Will raise an error if shape would cause memory overflow
    assert!(Tensor::<f32>::empty(&[i64::MAX, i64::MAX]).is_err());

    Ok(())
}
```