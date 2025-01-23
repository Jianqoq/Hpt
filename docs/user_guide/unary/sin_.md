# sin
```rust
Tensor::<T>::sin_(x: &Tensor<T>, out: &Tensor<T> | Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Trigonometric sine with out
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.sin_(&a)?;
    println!("{}", b);
    Ok(())
}
```