# abs
```rust
abs(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Calculate absolute of input
## Parameters:
`x`: input Tensor
## Returns:
Tensor with type `T`
## Examples:
```rust
use hpt::{ops::NormalUaryOps, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([-10.0]);
    let b = a.abs()?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |