# triangular
```rust
triangular(
    low: T,
    high: T,
    mode: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with values drawn from a triangular distribution. The triangular distribution is a continuous probability distribution with a lower limit `low`, upper limit `high`, and mode `mode`. It forms a triangular shape in its probability density function.

## Parameters:
`low`: Lower limit (a) of the distribution.

`high`: Upper limit (b) of the distribution. Must be greater than `low`.

`mode`: Mode (c) of the distribution. Must be between `low` and `high`.

`shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random values from the triangular distribution.

## Examples:
```rust
use hpt::{error::TensorError, ops::Random, Tensor};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with triangular distribution (low=0.0, high=10.0, mode=5.0)
    let a = Tensor::<f32>::triangular(0.0, 10.0, 5.0, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |