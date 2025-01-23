# asin
```rust
Tensor::<T>::asin(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Inverse sine
## Parameters:
`x`: Angle(radians)
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.asin()?;
    println!("{}", b);
    Ok(())
}
```