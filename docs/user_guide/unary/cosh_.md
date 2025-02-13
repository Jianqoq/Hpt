# cosh_
```rust
cosh_(
    x: &Tensor<T>, 
    out: &Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Hyperbolic cosine with out
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `C`
## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.cosh_(&a)?;
    println!("{}", b);
    Ok(())
}
```