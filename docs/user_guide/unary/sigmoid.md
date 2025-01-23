# sigmoid
```rust
Tensor::<T>::sigmoid(x: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large \frac{1}{1 + e^{-x}}$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `T`

## Examples:
```rust
use tensor_dyn::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.sigmoid()?;
    println!("{}", b);  // prints: 0.8807971
    Ok(())
}
```