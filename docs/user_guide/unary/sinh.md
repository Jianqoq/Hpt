# sinh
```rust
Tensor::<T>::sinh(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Hyperbolic sine
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.sinh()?;
    println!("{}", b);
    Ok(())
}
```