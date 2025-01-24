# acosh_
```rust
acosh_(
    x: &Tensor<T>, 
    out: &Tensor<C> | Tensor<C>
) -> Result<Tensor<C>, TensorError>
```
Inverse hyperbolic cosine with out
## Parameters:
`x`: Angle(radians)
`out`: Tensor to write to
## Returns:
Tensor with type `C`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.acosh_(&a)?;
    println!("{}", b);
    Ok(())
}
```