# cbrt
```rust
cbrt(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \sqrt[3]{x}$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([8.0]);
    let b = a.cbrt()?;
    println!("{}", b);  // prints: 2.0
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |