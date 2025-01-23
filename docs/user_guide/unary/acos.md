# acos
```rust
Tensor::<T>::acos(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Inverse cosine
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.acos()?;
    println!("{}", b);
    Ok(())
}
```