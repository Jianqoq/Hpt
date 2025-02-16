# gumbel_like
```rust
gumbel_like(
    x: &Tensor<T>,
    mu: T,
    beta: T
) -> Result<Tensor<T>, TensorError>
```
Same as `gumbel` but the shape will be based on `x`. Creates a Tensor with values drawn from a Gumbel distribution with location parameter `mu` and scale parameter `beta`.

## Parameters:
`x`: Input Tensor to derive the shape from

`mu`: Location parameter (μ) of the Gumbel distribution.

`beta`: Scale parameter (β) of the Gumbel distribution. Must be positive.

## Returns:
Tensor with type `T` containing random values from the Gumbel distribution.

## Examples:
```rust
use hpt::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with Gumbel distribution
    let g = x.gumbel_like(0.0, 1.0)?;
    println!("{}", g);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |