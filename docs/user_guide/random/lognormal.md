# lognormal
```rust
lognormal(
    mean: T,
    std: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a log-normal distribution. A random variable is log-normally distributed if the logarithm of the random variable is normally distributed. The parameters `mean` and `std` are the mean and standard deviation of the underlying normal distribution.

## Parameters:
`mean`: Mean (μ) of the underlying normal distribution.

`std`: Standard deviation (σ) of the underlying normal distribution. Must be positive.

`shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the log-normal distribution.

## Examples:
```rust
use hpt::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with log-normal distribution (μ=0.0, σ=1.0)
    let a = Tensor::<f32>::lognormal(0.0, 1.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```