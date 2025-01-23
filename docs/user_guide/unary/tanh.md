# tanh
```rust
Tensor::<T>::tanh(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Hyperbolic tangent
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.tanh()?;
    println!("{}", b);
    Ok(())
}
```