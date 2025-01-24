# acosh
```rust
acosh(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Inverse hyperbolic cosine
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `C`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.acosh()?;
    println!("{}", b);
    Ok(())
}
```