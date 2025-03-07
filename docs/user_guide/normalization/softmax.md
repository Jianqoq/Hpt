# dropout
```rust
dropout(
    x: &Tensor<T>,
    rate: f64
) -> Result<Tensor<T>, TensorError>
```
Randomly zeroes some of the elements of the input tensor with probability `rate` using samples from a Bernoulli distribution. Each element is zeroed independently.

## Parameters:
`x`: Input tensor.

`rate`: Probability of an element to be zeroed. The value must be between 0 and 1.

## Returns:
A new tensor with the same shape as input, where some elements are set to zero.

## Examples:
```rust
use hpt::{Tensor, TensorCreator, TensorError, AdvancedOps};

fn main() -> Result<(), TensorError> {
    let x = Tensor::<f32>::ones(&[3, 4])?;
    
    // Apply dropout with 0.5 probability
    let dropped = x.dropout(0.5)?;
    println!("After dropout:\n{}", dropped);
    
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |