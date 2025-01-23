# softplus
```rust
Tensor::<T>::softplus(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\ln(1 + e^x)$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.softplus()?;
    println!("{}", b);  // prints: 2.1269281
    Ok(())
}
```