# triangular_like
```rust
triangular_like(
    x: &Tensor<T>,
    low: T,
    high: T,
    mode: T
) -> Result<Tensor<T>, TensorError>
```
Same as `triangular` but the shape will be based on `x`. Creates a Tensor with values drawn from a triangular distribution with specified lower limit, upper limit, and mode.

## Parameters:
`x`: Input Tensor to derive the shape from

`low`: Lower limit (a) of the distribution.

`high`: Upper limit (b) of the distribution. Must be greater than `low`.

`mode`: Mode (c) of the distribution. Must be between `low` and `high`.

## Returns:
Tensor with type `T` containing random values from the triangular distribution.

## Examples:
```rust
use hpt::{error::TensorError, ops::Random, Tensor};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with triangular distribution
    let t = x.triangular_like(0.0, 10.0, 5.0)?;
    println!("{}", t);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |