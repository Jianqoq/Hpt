# gamma_like
```rust
gamma_like(
    x: &Tensor<T>,
    shape_param: T,
    scale: T
) -> Result<Tensor<T>, TensorError>
```
Same as `gamma` but the shape will be based on `x`. Creates a Tensor with values drawn from a gamma distribution with shape parameter `shape_param` and scale parameter `scale`.

## Parameters:
`x`: Input Tensor to derive the shape from

`shape_param`: Shape parameter (k or α) of the gamma distribution. Must be positive.

`scale`: Scale parameter (θ) of the gamma distribution. Must be positive.

## Returns:
Tensor with type `T` containing random values from the gamma distribution.

## Examples:
```rust
use hpt::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with gamma distribution
    let g = x.gamma_like(2.0, 2.0)?;
    println!("{}", g);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |