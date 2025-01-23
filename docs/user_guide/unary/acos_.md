# acos
```rust
Tensor::<T>::acos_(
    x: &Tensor<T>, 
    out: &Tensor<T> | Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Inverse cosine with out
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
    let b = a.acos_(&a)?;
    println!("{}", b);
    Ok(())
}
```