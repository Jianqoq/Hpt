# elu
```rust
elu(x: &Tensor<T>, alpha: C) -> Result<Tensor<C>, TensorError>
```
Compute $\large x$ for $x > 0$, $\large \alpha(e^x - 1)$ for $x \leq 0$ for all elements

## Parameters:
`x`: Input values
`alpha`: Parameter controlling the saturation of negative values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{ops::FloatUnaryOps, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([-1.0]);
    let b = a.elu(1.0)?;
    println!("{}", b);  // prints: -0.6321206
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |