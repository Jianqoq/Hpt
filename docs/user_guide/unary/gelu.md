# gelu
```rust
gelu(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\large x \cdot \Phi(x)$ where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{ops::FloatUnaryOps, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.gelu()?;
    println!("{}", b);  // prints: 1.9545977
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |