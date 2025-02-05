# chisquare_like
```rust
chisquare_like(
    x: &Tensor<T>,
    df: T
) -> Result<Tensor<T>, TensorError>
```
Same as `chisquare` but the shape will be based on `x`. Creates a Tensor with values drawn from a chi-square distribution with `df` degrees of freedom.

## Parameters:
`x`: Input Tensor to derive the shape from

`df`: Degrees of freedom parameter. Must be positive.

## Returns:
Tensor with type `T` containing random values from the chi-square distribution.

## Examples:
```rust
use hpt_core::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with chi-square distribution
    let c = x.chisquare_like(5.0)?;
    println!("{}", c);
    Ok(())
}
```