# randint_like
```rust
randint_like(
    x: &Tensor<T>,
    low: T,
    high: T
) -> Result<Tensor<T>, TensorError>
```
Same as `randint` but the shape will be based on `x`. Creates a Tensor with random integers drawn uniformly from the half-open interval `[low, high)`.

## Parameters:
`x`: Input Tensor to derive the shape from

`low`: Lower bound (inclusive) of the range.

`high`: Upper bound (exclusive) of the range.

## Returns:
Tensor with type `T` containing random integers in the range `[low, high)`.

## Examples:
```rust
use hpt::{RandomInt, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<i32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with random integers
    let r = x.randint_like(0, 100)?;
    println!("{}", r);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |