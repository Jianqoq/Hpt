# sinh
```rust
sinh(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Hyperbolic sine
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.sinh()?;
    println!("{}", b);
    Ok(())
}
```