# exp
```rust
Tensor::<T>::exp(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute exponential of `x` for all elements
## Parameters:
`x`: Input values
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.exp()?;
    println!("{}", b);
    Ok(())
}
```