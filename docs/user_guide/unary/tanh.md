# tanh
```rust
tanh(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Hyperbolic tangent
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt_core::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.tanh()?;
    println!("{}", b);
    Ok(())
}
```