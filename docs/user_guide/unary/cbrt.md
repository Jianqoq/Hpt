# cbrt
```rust
Tensor::<T>::cbrt(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large \sqrt[3]{x}$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([8.0]);
    let b = a.cbrt()?;
    println!("{}", b);  // prints: 2.0
    Ok(())
}
```