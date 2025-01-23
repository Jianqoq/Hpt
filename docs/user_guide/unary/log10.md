# log10
```rust
Tensor::<T>::log10(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large \log_{10}(x)$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([100.0]);
    let b = a.log10()?;
    println!("{}", b);
    Ok(())
}
```