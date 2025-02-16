# beta_like
```rust
beta_like(
    x: &Tensor<T>,
    alpha: T,
    beta: T
) -> Result<Tensor<T>, TensorError>
```
Same as `beta` but the shape will be based on `x`. Creates a Tensor with values drawn from a beta distribution with parameters `alpha` and `beta`.

## Parameters:
`x`: Input Tensor to derive the shape from

`alpha`: Shape parameter alpha (α) of the beta distribution. Must be positive.

`beta`: Shape parameter beta (β) of the beta distribution. Must be positive.

## Returns:
Tensor with type `T` containing random values from the beta distribution

## Examples:
```rust
use hpt::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with beta distribution
    let b = x.beta_like(2.0, 5.0)?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |