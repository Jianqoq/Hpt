# expand
```rust
expand(
    x: &Tensor<T>,
    shape: 
        &[i64]
        | &[i64; _]
        | [i64; _] 
        | Vec<i64> 
        | &Vec<i64>
) -> Result<Tensor<T>, TensorError>
```
Expands the tensor to a larger size, replicating the data along specified dimensions.

## Parameters:
`x`: Input tensor

`shape`: The desired expanded shape. Must be compatible with the input tensor's shape, where each dimension must either be equal to the input dimension or the input dimension must be 1.

## Returns:
A new tensor with expanded dimensions.

## Examples:
```rust
use tensor_dyn::{ShapeManipulate, Tensor, TensorCreator, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // Create a tensor with shape [1, 3, 1]
    let a = Tensor::<f32>::zeros(&[1, 3, 1])?;

    // Expand to shape [2, 3, 4]
    let b = a.expand(&[2, 3, 4])?;
    println!("{}", b.shape());

    // This will return an error as we can't expand dimension 1 from 3 to 4
    let c = a.expand(&[2, 4, 4]);
    assert!(c.is_err());

    Ok(())
}
```