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
use hpt::{ops::FloatUnaryOps, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.acosh()?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |