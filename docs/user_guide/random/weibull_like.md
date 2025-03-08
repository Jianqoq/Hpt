# weibull_like
```rust
weibull_like(
    x: &Tensor<T>,
    shape: T,
    scale: T
) -> Result<Tensor<T>, TensorError>
```
Same as `weibull` but the shape will be based on `x`. Creates a Tensor with values drawn from a Weibull distribution with specified shape and scale parameters.

## Parameters:
`x`: Input Tensor to derive the shape from

`shape`: Shape parameter (k), determines the shape of the distribution. Must be positive.

`scale`: Scale parameter (λ), determines the spread of the distribution. Must be positive.

## Returns:
Tensor with type `T` containing random values from the Weibull distribution.

## Examples:
```rust
use hpt::{error::TensorError, ops::Random, Tensor};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with Weibull distribution
    let w = x.weibull_like(2.0, 1.0)?;
    println!("{}", w);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |