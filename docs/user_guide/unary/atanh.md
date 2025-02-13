# atanh
```rust
atanh(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Inverse hyperbolic tangent
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.atanh()?;
    println!("{}", b);
    Ok(())
}
```