# exp2
```rust
Tensor::<T>::exp2(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large 2^x$ for all elements
## Parameters:
`x`: Input values
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.exp2()?;
    println!("{}", b);
    Ok(())
}
```