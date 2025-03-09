# transpose
```rust
transpose(
    x: &Tensor<T>,
    axis1: i64,
    axis2: i64
) -> Result<Tensor<T>, TensorError>
```
Swaps two axes of the 2D tensor, returning a view of the tensor with axes transposed.

## Parameters:
`x`: Input tensor

`axis1`: First axis to be transposed

`axis2`: Second axis to be transposed

## Returns:
A new tensor with the specified axes transposed.

## Examples:
```rust
use hpt::{
    common::TensorInfo,
    error::TensorError,
    ops::{ShapeManipulate, TensorCreator},
    Tensor,
};
fn main() -> Result<(), TensorError> {
    // Create a tensor with shape [2, 4]
    let a = Tensor::<f32>::zeros(&[2, 4])?;

    let b = a.transpose(0, 1)?; // shape becomes [2, 4]
    println!("{}", b.shape());

    let c = a.transpose(1, 0)?; // shape becomes [4, 2]
    println!("{}", c.shape());

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |