# zipf_like
```rust
zipf_like(
    x: &Tensor<T>,
    n: T,
    s: T
) -> Result<Tensor<T>, TensorError>
```
Same as `zipf` but the shape will be based on `x`. Creates a Tensor with values drawn from a Zipf distribution with specified number of elements and exponent parameter.

## Parameters:
`x`: Input Tensor to derive the shape from

`n`: Number of elements (N). Defines the range of possible values [1, N].

`s`: Exponent parameter (s). Controls the skewness of the distribution. Must be greater than 1.

## Returns:
Tensor with type `T` containing random values from the Zipf distribution.

## Examples:
```rust
use hpt::{error::TensorError, ops::Random, Tensor};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with Zipf distribution
    let z = x.zipf_like(1000.0, 2.0)?;
    println!("{}", z);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |