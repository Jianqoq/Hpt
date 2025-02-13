# atan
```rust
atan(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Inverse tangent
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.atan()?;
    println!("{}", b);
    Ok(())
}
```