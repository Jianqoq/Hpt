# sqrt
```rust
sqrt(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\sqrt{x}$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{ops::FloatUnaryOps, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([4.0]);
    let b = a.sqrt()?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |