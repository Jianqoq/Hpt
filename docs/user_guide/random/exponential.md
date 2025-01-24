# exponential
```rust
exponential(
    lambda: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from an exponential distribution with rate parameter `lambda`. The exponential distribution describes the time between events in a Poisson point process.

## Parameters:
`lambda`: Rate parameter (λ) of the exponential distribution. Must be positive.

`shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the exponential distribution.

## Examples:
```rust
use tensor_dyn::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with exponential distribution (λ=2.0)
    let a = Tensor::<f32>::exponential(2.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```