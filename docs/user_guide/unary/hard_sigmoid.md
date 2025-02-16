# hard_sigmoid
```rust
hard_sigmoid(x: &Tensor<T>) -> Result<Tensor<C>, TensorError>
```
Compute $\large \text{max}(0, \text{min}(1, \frac{x}{6} + 0.5))$ for all elements. A piece-wise linear approximation of the sigmoid function.

## Parameters:
`x`: Input values

## Returns:
Tensor with type `C`

## Examples:
```rust
use hpt::{FloatUnaryOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::new([2.0]);
    let b = a.hard_sigmoid()?;
    println!("{}", b);  // prints: 0.8333333
    Ok(())
}
```
## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ✅        |