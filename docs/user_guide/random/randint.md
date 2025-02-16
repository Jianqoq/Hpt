# randint
```rust
randint(
    low: T,
    high: T,
    shape: &[i64] | &Vec<i64> | &[i64; _]
) -> Result<Tensor<T>, TensorError>
```
Create a Tensor with random integers drawn uniformly from the half-open interval `[low, high)`. The distribution is uniform, meaning each integer in the range has an equal probability of being drawn.

## Parameters:
`low`: Lower bound (inclusive) of the range.

`high`: Upper bound (exclusive) of the range.

`shape`: Shape of the output tensor.

## Returns:
Tensor with type `T` containing random integers in the range `[low, high)`.

## Examples:
```rust
use hpt::{RandomInt, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create a 10x10 tensor with random integers between 0 and 100
    let a = Tensor::<i32>::randint(0, 100, &[10, 10])?;
    println!("{}", a);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |