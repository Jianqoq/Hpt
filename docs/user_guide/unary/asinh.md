# asinh
```rust
Tensor::<T>::asinh(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Inverse hyperbolic sine
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.asinh()?;
    println!("{}", b);
    Ok(())
}
```