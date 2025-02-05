# cosh
```rust
cosh(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Hyperbolic cosine
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.cosh()?;
    println!("{}", b);
    Ok(())
}
```