# sigmoid
```rust
sigmoid(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \frac{1}{1 + e^{-x}}$ for all elements

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

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