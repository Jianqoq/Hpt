# poisson
```rust
poisson(
    lambda: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a Poisson distribution. The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space, assuming these events occur with a known constant mean rate (λ) and independently of the time since the last event.

## Parameters:
`lambda`: Rate parameter (λ) of the Poisson distribution. Must be positive.

`shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the Poisson distribution.

## Examples:
```rust
use hpt_core::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with Poisson distribution (λ=5.0)
    let a = Tensor::<f32>::poisson(5.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```