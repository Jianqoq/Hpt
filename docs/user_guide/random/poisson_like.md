# poisson_like
```rust
poisson_like(
    x: &Tensor<T>,
    lambda: T
) -> Result<Tensor<T>, TensorError>
```
Same as `poisson` but the shape will be based on `x`. Creates a Tensor with values drawn from a Poisson distribution with specified rate parameter.

## Parameters:
`x`: Input Tensor to derive the shape from

`lambda`: Rate parameter (λ) of the Poisson distribution. Must be positive.

## Returns:
Tensor with type `T` containing random values from the Poisson distribution.

## Examples:
```rust
use hpt::{error::TensorError, ops::Random, Tensor};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with Poisson distribution
    let p = x.poisson_like(5.0)?;
    println!("{}", p);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |