# tan_
```rust
Tensor::<T>::tan_(
    x: &Tensor<T>, 
    out: &Tensor<T> | Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Trigonometric tangent with out
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.tan_(&a)?;
    println!("{}", b);
    Ok(())
}
```