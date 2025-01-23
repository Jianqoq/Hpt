# tan
```rust
Tensor::<T>::tan(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Trigonometric tangent
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.tan()?;
    println!("{}", b);
    Ok(())
}
```