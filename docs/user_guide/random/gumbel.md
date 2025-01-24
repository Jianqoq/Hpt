# gumbel
```rust
gumbel(
    mu: T,
    beta: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a Gumbel distribution (also known as the Extreme Value Type I distribution) with location parameter `mu` and scale parameter `beta`. The Gumbel distribution is commonly used to model the distribution of extreme values.

## Parameters:
`mu`: Location parameter (μ) of the Gumbel distribution.

`beta`: Scale parameter (β) of the Gumbel distribution. Must be positive.

`shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the Gumbel distribution.

## Examples:
```rust
use tensor_dyn::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with Gumbel distribution (μ=0.0, β=1.0)
    let a = Tensor::<f32>::gumbel(0.0, 1.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```