# log2
```rust
Tensor::<T>::log2(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large \log_{2}(x)$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([8.0]);
    let b = a.log2()?;
    println!("{}", b);  // prints: 3.0
    Ok(())
}
```