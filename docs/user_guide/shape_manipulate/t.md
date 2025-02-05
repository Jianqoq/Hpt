# t
```rust
t(
    x: &Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Transposes the tensor by swapping the last two dimensions. For 1D or 2D tensors, this is equivalent to a regular transpose. For higher dimensional tensors, only the last two dimensions are swapped.

## Parameters:
`x`: Input tensor

## Returns:
A new tensor with its last two dimensions transposed.

## Examples:
```rust
use hpt_core::{ShapeManipulate, Tensor, TensorCreator, TensorError, TensorInfo};
fn main() -> Result<(), TensorError> {
    // 2D tensor example
    let a = Tensor::<f32>::zeros(&[2, 3])?;
    let b = a.t()?; // shape becomes [3, 2]
    println!("{}", b.shape());

    // 3D tensor example
    let c = Tensor::<f32>::zeros(&[2, 3, 4])?;
    let d = c.t()?; // shape becomes [2, 4, 3]
    println!("{}", d.shape());

    // 4D tensor example
    let e = Tensor::<f32>::zeros(&[2, 3, 4, 5])?;
    let f = e.t()?; // shape becomes [2, 3, 5, 4]
    println!("{}", f.shape());

    Ok(())
}
```