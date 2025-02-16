# pareto_like
```rust
pareto_like(
    x: &Tensor<T>,
    scale: T,
    shape: T
) -> Result<Tensor<T>, TensorError>
```
Same as `pareto` but the shape will be based on `x`. Creates a Tensor with values drawn from a Pareto distribution with specified scale and shape parameters.

## Parameters:
`x`: Input Tensor to derive the shape from

`scale`: Scale parameter (xₘ), also known as the minimum possible value. Must be positive.

`shape`: Shape parameter (α), also known as the Pareto index. Must be positive.

## Returns:
Tensor with type `T` containing random values from the Pareto distribution.

## Examples:
```rust
use hpt::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with Pareto distribution
    let p = x.pareto_like(1.0, 3.0)?;
    println!("{}", p);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |