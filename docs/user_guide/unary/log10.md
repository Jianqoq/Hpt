# log10
```rust
log10(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \log_{10}(x)$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([100.0]);
    let b = a.log10()?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |