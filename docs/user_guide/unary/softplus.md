# softplus
```rust
softplus(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\ln(1 + e^x)$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.softplus()?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |