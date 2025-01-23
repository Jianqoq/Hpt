# ln
```rust
Tensor::<T>::ln(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large \ln(x)$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.718281828459045]);
    let b = a.ln()?;
    println!("{}", b);  // prints: 1.0
    Ok(())
}
```