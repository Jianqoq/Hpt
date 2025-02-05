# acos
```rust
acos(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Inverse cosine
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.acos()?;
    println!("{}", b);
    Ok(())
}
```