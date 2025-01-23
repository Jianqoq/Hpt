# atan
```rust
Tensor::<T>::atan_(
    x: &Tensor<T>, 
    out: &Tensor<T> | Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Inverse tangent with out
## Parameters:
`x`: Angle(radians)
`out`: Tensor to write to
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.atan_(&a)?;
    println!("{}", b);
    Ok(())
}
```