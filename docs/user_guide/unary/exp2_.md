# exp2_
```rust
Tensor::<T>::exp2_(
    x: &Tensor<T>, 
    out: &Tensor<T> | Tensor<T>
) -> Result<Tensor<T>, TensorError>
```
Compute $\large 2^x$ for all elements with out
## Parameters:
`x`: Input values
## Returns:
Tensor with type `T`
## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([10.0]);
    let b = a.exp2_(&a)?;
    println!("{}", b);
    Ok(())
}
```