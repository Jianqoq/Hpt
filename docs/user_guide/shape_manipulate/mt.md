# mt
```rust
mt(
    x: &Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Performs a complete transpose by reversing all dimensions of the tensor. This is different from `t()` which only swaps the last two dimensions.

## Parameters:
`x`: Input tensor

## Returns:
A new tensor with all its dimensions reversed.

## Examples:
```rust
use hpt::{
    common::TensorInfo,
    error::TensorError,
    ops::{ShapeManipulate, TensorCreator},
    Tensor,
};
fn main() -> Result<(), TensorError> {
    // 2D tensor example
    let a = Tensor::<f32>::zeros(&[2, 3])?;
    let b = a.mt()?; // shape becomes [3, 2]
    println!("{}", b.shape());

    // 3D tensor example
    let c = Tensor::<f32>::zeros(&[2, 3, 4])?;
    let d = c.mt()?; // shape becomes [4, 3, 2]
    println!("{}", d.shape());

    // 4D tensor example
    let e = Tensor::<f32>::zeros(&[2, 3, 4, 5])?;
    let f = e.mt()?; // shape becomes [5, 4, 3, 2]
    println!("{}", f.shape());

    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |