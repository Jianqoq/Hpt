# mish
```rust
mish(x: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large x * \tanh(\ln(1 + e^x))$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{ops::FloatUnaryOps, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.mish()?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |