# gamma
```rust
gamma(
    shape_param: T,
    scale: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a gamma distribution with shape parameter `shape_param` (often denoted as k or α) and scale parameter `scale` (often denoted as θ). The gamma distribution is a continuous probability distribution that generalizes the exponential distribution.

## Parameters:
`shape_param`: Shape parameter (k or α) of the gamma distribution. Must be positive.

`scale`: Scale parameter (θ) of the gamma distribution. Must be positive.

`shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the gamma distribution.

## Examples:
```rust
use tensor_dyn::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with gamma distribution (k=2.0, θ=2.0)
    let a = Tensor::<f32>::gamma(2.0, 2.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```