# atan
```rust
Tensor::<T>::atan(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Inverse tangent
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.atan()?;
    println!("{}", b);
    Ok(())
}
```