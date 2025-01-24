# unsqueeze
```rust
unsqueeze(
    axes: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | i64
) -> Result<Tensor<T>, TensorError>
```
Adds a new dimension of size 1 to the tensor at the specified dimention.

## Parameters:
`x`: Input tensor

`axes`: The positions where the single-dimensional entries should be add.

## Returns:
A new tensor with the specified single-dimensional entries added.

## Examples:
```rust
use tensor_dyn::{ShapeManipulate, Tensor, TensorCreator, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create a tensor with shape [3, 4]
    let a = Tensor::<f32>::zeros(&[3, 4])?;

    // Remove single-dimensional entry at axis 0
    let b = a.unsqueeze(0)?; // shape becomes [1, 3, 4]
    println!("{}", b.shape());
    // Remove single-dimensional entry at axis 2
    let c = a.unsqueeze(1)?; // shape becomes [3, 1, 4]
    println!("{}", c.shape());

    Ok(())
}
```