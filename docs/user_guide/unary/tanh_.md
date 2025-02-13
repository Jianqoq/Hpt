# tanh_
```rust
tanh_(
    x: &Tensor<T>, 
    out: &Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Hyperbolic tangent with out
## Parameters:
`x`: Angle(radians)
`out`: Tensor to write to
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.tanh_(&a)?;
    println!("{}", b);
    Ok(())
}
```