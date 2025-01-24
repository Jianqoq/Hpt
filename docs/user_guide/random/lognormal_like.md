# lognormal_like
```rust
lognormal_like(
    x: &Tensor<T>,
    mean: T,
    std: T
) -> Result<Tensor<T>, TensorError>
```
Same as `lognormal` but the shape will be based on `x`. Creates a Tensor with values drawn from a log-normal distribution with parameters `mean` and `std` of the underlying normal distribution.

## Parameters:
`x`: Input Tensor to derive the shape from

`mean`: Mean (μ) of the underlying normal distribution.

`std`: Standard deviation (σ) of the underlying normal distribution. Must be positive.

## Returns:
Tensor with type `T` containing random values from the log-normal distribution.

## Examples:
```rust
use tensor_dyn::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with log-normal distribution
    let l = x.lognormal_like(0.0, 1.0)?;
    println!("{}", l);
    Ok(())
}
```