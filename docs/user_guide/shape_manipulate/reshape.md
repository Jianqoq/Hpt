# reshape
```rust
reshape(
    shape: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
) -> Result<Tensor<T>, TensorError>
```
Gives a new shape to the tensor without changing its data.

## Parameters:
`x`: Input tensor

`shape`: The new shape. The total number of elements must remain the same.

## Returns:
A new tensor with the same data but reshaped to the specified dimensions.

## Examples:
```rust
use hpt::{
    common::TensorInfo,
    error::TensorError,
    ops::{ShapeManipulate, TensorCreator},
    Tensor,
};
fn main() -> Result<(), TensorError> {
    // Create a tensor with shape [3, 4]
    let a = Tensor::<f32>::zeros(&[3, 4])?;

    // Reshape to [2, 6]
    let b = a.reshape(&[2, 6])?;
    println!("{}", b.shape());

    // Reshape to [12]
    let c = a.reshape(&[12])?;
    println!("{}", c.shape());

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |