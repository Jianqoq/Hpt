# exponential_like
```rust
exponential_like(
    x: &Tensor<T>,
    lambda: T
) -> Result<Tensor<T>, TensorError>
```
Same as `exponential` but the shape will be based on `x`. Creates a Tensor with values drawn from an exponential distribution with rate parameter `lambda`.

## Parameters:
`x`: Input Tensor to derive the shape from

`lambda`: Rate parameter (λ) of the exponential distribution. Must be positive.

## Returns:
Tensor with type `T` containing random values from the exponential distribution.

## Examples:
```rust
use tensor_dyn::{Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Create an initial tensor
    let x = Tensor::<f32>::randn(&[10, 10])?;
    
    // Create a new tensor with same shape as x but with exponential distribution
    let e = x.exponential_like(2.0)?;
    println!("{}", e);
    Ok(())
}
```