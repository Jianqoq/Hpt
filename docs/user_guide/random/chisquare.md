# chisquare
```rust
chisquare(
    df: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a chi-square distribution with `df` degrees of freedom. The chi-square distribution is a continuous probability distribution of the sum of squares of `df` independent standard normal random variables.

## Parameters:
`df`: Degrees of freedom parameter. Must be positive.

`shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the chi-square distribution.

## Examples:
```rust
use tensor_dyn::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with chi-square distribution (df=5)
    let a = Tensor::<f32>::chisquare(5.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```