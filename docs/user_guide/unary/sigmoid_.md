# sigmoid_
```rust
Tensor::<T>::sigmoid_(x: &Tensor<T>, out: &Tensor<T>) -> Result<Tensor<T>, TensorError>
```
Compute $\large \frac{1}{1 + e^{-x}}$ for all elements with out

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
    let b = a.sigmoid_(&a)?;
    println!("{}", b);  // prints: 0.8807971
    Ok(())
}
```