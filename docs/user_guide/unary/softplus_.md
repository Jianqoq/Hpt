# softplus_
```rust
Tensor::<T>::softplus_(x: &Tensor<T>, out: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\ln(1 + e^x)$ for all elements with out

## Parameters:
`x`: Input values
`out`: Tensor to write to

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.softplus_(&a)?;
    println!("{}", b);  // prints: 2.1269281
    Ok(())
}
```