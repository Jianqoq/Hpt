# recip
```rust
recip(x: &Tensor<C>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \frac{1}{x}$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.recip()?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |