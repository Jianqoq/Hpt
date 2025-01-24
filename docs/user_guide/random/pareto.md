# pareto
```rust
pareto(
    scale: T,
    shape: T,
    tensor_shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a Pareto distribution. The Pareto distribution is a power-law probability distribution often used to describe the distribution of wealth, population sizes, and many other natural and social phenomena.

## Parameters:
`scale`: Scale parameter (xₘ), also known as the minimum possible value. Must be positive.

`shape`: Shape parameter (α), also known as the Pareto index. Must be positive.

`tensor_shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the Pareto distribution.

## Examples:
```rust
use tensor_dyn::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with Pareto distribution (scale=1.0, shape=3.0)
    let a = Tensor::<f32>::pareto(1.0, 3.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```