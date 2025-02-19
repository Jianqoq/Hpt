# asinh
```rust
asinh(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Inverse hyperbolic sine
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.asinh()?;
    println!("{}", b);
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |