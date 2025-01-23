# sqrt
```rust
Tensor::<T>::sqrt(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\sqrt{x}$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([4.0]);
    let b = a.sqrt()?;
    println!("{}", b);  // prints: 2.0
    Ok(())
}
```