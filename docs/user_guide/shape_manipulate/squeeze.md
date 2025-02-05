# squeeze
```rust
squeeze(
    axes: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
        | i64
) -> Result<Tensor<T>, TensorError>
```
Remove single-dimensional entries (axes with size 1) from the shape of the tensor at specified positions.

## Parameters:
`axes`: The positions where the single-dimensional entries should be removed.

## Returns:
A new tensor with the specified single-dimensional entries removed.

## Examples:
```rust
use hpt_core::{ShapeManipulate, Tensor, TensorCreator, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create a tensor with shape [1, 3, 1, 4]
    let a = Tensor::<f32>::zeros(&[1, 3, 1, 4])?;

    // Remove single-dimensional entry at axis 0
    let b = a.squeeze(0)?; // shape becomes [3, 1, 4]
    println!("{}", b.shape());
    // Remove single-dimensional entry at axis 2
    let c = a.squeeze(2)?; // shape becomes [1, 3, 4]
    println!("{}", c.shape());

    Ok(())
}

```