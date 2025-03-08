# ln
```rust
ln(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \ln(x)$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{ops::FloatUnaryOps, Tensor, error::TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.718281828459045]);
    let b = a.ln()?;
    println!("{}", b);  // prints: 1.0
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |