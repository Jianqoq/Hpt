# dropout
```rust
fn dropout(
    x: &Tensor<T>, 
    rate: f64
) -> Result<Tensor<T>, TensorError>
```
Randomly zeroes some of the elements of the input tensor with probability rate using samples from a Bernoulli distribution. Each element is zeroed independently.

## Parameters:
`x`: Input tensor.

`rate`: Probability of an element to be zeroed. The value must be between 0 and 1.

## Returns:
Tensor with type `T`

## Examples:
```rust
use hpt::{
    error::TensorError,
    ops::{RegularizationOps, TensorCreator},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // Create a tensor filled with ones
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