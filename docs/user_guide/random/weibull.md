# weibull
```rust
weibull(
    shape: T,
    scale: T,
    tensor_shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a Weibull distribution. The Weibull distribution is a continuous probability distribution commonly used in reliability engineering, survival analysis, and extreme value theory.

## Parameters:
`shape`: Shape parameter (k), determines the shape of the distribution. Must be positive.

`scale`: Scale parameter (λ), determines the spread of the distribution. Must be positive.

`tensor_shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the Weibull distribution.

## Examples:
```rust
use hpt_core::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with Weibull distribution (shape=2.0, scale=1.0)
    let a = Tensor::<f32>::weibull(2.0, 1.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```